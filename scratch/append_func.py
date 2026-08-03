
def re_synthesize_company_data(org_id: int):
    """Re-synthesize data using ONLY the currently saved sources, bypassing Exa API calls."""
    db = SessionLocal()
    from market_comps.db.models import EntityMatch, ExtractedEntity, ExtractionJob, DocumentText
    try:
        org = db.query(Organization).filter_by(id=org_id).first()
        if not org:
            raise ValueError("Organization not found")
            
        # Get existing docs
        doc_ids_subquery_1 = db.query(SourceDocument.id).join(DocumentText).join(ExtractionJob).join(ExtractedEntity).join(EntityMatch).filter(
            EntityMatch.canonical_entity_type == "Organization",
            EntityMatch.canonical_entity_id == str(org.id)
        ).subquery()
        
        doc_ids_subquery_2 = db.query(SourceDocument.id).join(PipelineRun, SourceDocument.pipeline_run_id == PipelineRun.id).join(CompanyAugmentationReport, PipelineRun.id == CompanyAugmentationReport.pipeline_run_id).filter(
            CompanyAugmentationReport.organization_id == org.id
        ).subquery()
        
        docs = db.query(SourceDocument).filter(
            (SourceDocument.id.in_(doc_ids_subquery_1)) | (SourceDocument.id.in_(doc_ids_subquery_2))
        ).all()
        
        # Deduplicate
        seen_urls = set()
        deduped_docs = []
        for doc in docs:
            url = str(doc.source_url).strip().lower()
            if url.endswith('/'): url = url[:-1]
            if url not in seen_urls:
                seen_urls.add(url)
                deduped_docs.append(doc)
                
        if not deduped_docs:
            raise ValueError("No source documents found to re-synthesize.")
            
        docs_data = []
        for d in deduped_docs:
            txt = db.query(DocumentText).filter_by(source_document_id=d.id).first()
            docs_data.append({
                "db_id": d.id,
                "url": d.source_url,
                "title": d.title,
                "date": d.document_date.strftime("%Y-%m-%d") if isinstance(d.document_date, datetime) else str(d.document_date) if d.document_date else None,
                "text": txt.raw_content if txt else ""
            })
            
        # Create Report
        report = CompanyAugmentationReport(
            organization_id=org_id,
            schema_version="1.0",
            status="RUNNING"
        )
        db.add(report)
        db.commit()
        
        # Create Pipeline Run
        run = PipelineRun(
            pipeline_id=None,
            run_status="IN_PROGRESS"
        )
        db.add(run)
        db.commit()
        
        report.pipeline_run_id = run.id
        db.commit()
        
        # Initialize usage trackers
        if not run.llm_total_tokens: run.llm_total_tokens = 0
        if not run.llm_estimated_cost_usd: run.llm_estimated_cost_usd = 0.0
        if not run.exa_calls: run.exa_calls = 0
        if not run.exa_estimated_cost_usd: run.exa_estimated_cost_usd = 0.0
        
        def add_usage(u):
            if u:
                run.llm_total_tokens += u.total_tokens
                run.llm_estimated_cost_usd += u.estimated_cost_usd
        
        # 3 & 4. Process and Score
        extracted_data, u_score = process_and_score_evidence(docs_data)
        add_usage(u_score)
        
        scoring = {}
        for k, v in extracted_data.items():
            if k == "executive_summary":
                continue
            scoring[k] = {
                "score": v.get("score_1_to_10"),
                "confidence": v.get("confidence"),
                "reasoning": v.get("reasoning")
            }
            
        report.extracted_data_json = extracted_data
        report.scoring_json = scoring
        report.status = "SUCCESS"
        
        # 5. Extract and Upsert Basics
        basics, u_basics = extract_company_basics(docs_data)
        add_usage(u_basics)
        if basics.get("website"): org.website_url = basics["website"]
        if basics.get("description_short"): org.description = basics["description_short"]
        
        loc = basics.get("hq_location")
        if loc:
            parts = [p.strip() for p in loc.split(",")]
            if len(parts) >= 1: org.city = parts[0]
            if len(parts) >= 2: org.state = parts[1]
            if len(parts) >= 3: org.country = parts[2]
            
        profile = db.query(CompanyProfile).filter_by(organization_id=org.id).first()
        if not profile:
            profile = CompanyProfile(organization_id=org.id)
            db.add(profile)
        
        if basics.get("founded_year"): profile.founded_year = basics["founded_year"]
        if basics.get("sector"): profile.industry = basics["sector"]
        if basics.get("subsector"): profile.subindustry = basics["subsector"]
        
        db.flush()

        # 6. Extract and Upsert Entities (Founders / Team)
        people, u_people = extract_entities(docs_data)
        add_usage(u_people)
        for p in people:
            first = p.get("first_name")
            last = p.get("last_name")
            if not first or not last: continue
            
            full_name = f"{first} {last}"
            person = db.query(Person).filter(Person.full_name.ilike(full_name)).first()
            if not person:
                person = Person(
                    first_name=first,
                    last_name=last,
                    full_name=full_name,
                    city=p.get("city"),
                    linkedin_url=p.get("linkedin_url"),
                    country="US"
                )
                db.add(person)
                db.flush()
            
            if p.get("email"):
                existing_email = db.query(PersonEmail).filter_by(person_id=person.id, email=p["email"]).first()
                if not existing_email:
                    email_record = PersonEmail(
                        person_id=person.id,
                        email=p["email"],
                        organization_id=org.id,
                        is_primary=True
                    )
                    db.add(email_record)
            
            title = p.get("title") or ("Founder" if p.get("is_founder") else "Executive")
            role = db.query(PersonOrganizationRole).filter_by(person_id=person.id, organization_id=org.id).first()
            if not role:
                role = PersonOrganizationRole(
                    person_id=person.id,
                    organization_id=org.id,
                    title=title,
                    source="WEB_AUGMENTATION"
                )
                db.add(role)
                db.flush()
            elif p.get("is_founder") and "founder" not in (role.title or "").lower():
                role.title = "Founder & " + (role.title or "Executive")
        
        # 7. Extract and Upsert Investments
        investments_data, u_inv = extract_investments(docs_data)
        add_usage(u_inv)
        for inv in investments_data:
            investor_name = inv.get("investor_name")
            if not investor_name: continue
            
            investor_org = db.query(Organization).filter(Organization.name.ilike(investor_name)).first()
            if not investor_org:
                investor_org = Organization(
                    name=investor_name,
                    organization_type="INVESTOR",
                    status="ACTIVE"
                )
                db.add(investor_org)
                db.flush()
            
            doc_idx = inv.get("source_doc_index")
            source_doc_id = None
            if doc_idx is not None and 0 <= doc_idx < len(docs_data):
                source_doc_id = docs_data[doc_idx].get("db_id")
                
            investment = Investment(
                investor_organization_id=investor_org.id,
                company_organization_id=org.id,
                round_type=inv.get("round_type"),
                total_round_amount=inv.get("total_round_amount"),
                firm_investment_amount=inv.get("firm_investment_amount"),
                is_lead=inv.get("is_lead", False),
                source_document_id=source_doc_id
            )
            
            date_str = inv.get("investment_date")
            if date_str:
                try:
                    investment.investment_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    pass
            
            db.add(investment)
            db.flush()
            
        db.commit()
        run.run_status = "SUCCESS"
        run.completed_at = datetime.utcnow()
        db.commit()
        
    except Exception as e:
        logger.error(f"Re-synthesis pipeline failed: {e}")
        if 'report' in locals():
            report.status = "FAILED"
            report.error_message = str(e)
            db.commit()
        raise e
    finally:
        db.close()
