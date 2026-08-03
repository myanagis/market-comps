import importlib.util
import sys
spec = importlib.util.spec_from_file_location('company_mapper', 'pages/19_Company_Name_Mapper.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['company_mapper'] = mod
spec.loader.exec_module(mod)

df = mod.map_companies([
    'Smart Feed Tech Inc.',
    'Smart Seal Inc.',
    'Skyview Ventures',
    'Spice Ventures'
])
for _, row in df.iterrows():
    print(str(row['original_name']) + ' -> ' + str(row['status']) + ' (' + str(row['score']) + ') -> ' + str(row['reason']))
