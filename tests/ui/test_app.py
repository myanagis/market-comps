import pytest
from streamlit.testing.v1 import AppTest

def test_app_startup():
    """
    Test that the main Streamlit application starts without raising exceptions.
    AppTest allows running the Streamlit app headlessly to verify rendering.
    """
    # Create the test runner for app.py
    at = AppTest.from_file("app.py")
    
    # Run the app. Setting timeout just in case it hangs.
    at.run(timeout=30)
    
    # Verify the app did not throw an unhandled exception
    assert not at.exception, f"App raised an exception: {at.exception[0]}"
    
    # Check that the basic layout or expected static text rendered
    assert "🔑 Admin Login" in [exp.label for exp in at.expander], "Expected Admin Login expander to exist on the sidebar"

def test_bypass_login_renders_sidebar():
    """
    Test that clicking the dev bypass login successfully updates session state
    and displays the logged-in sidebar view.
    """
    at = AppTest.from_file("app.py")
    at.run()
    
    # Verify we start logged out
    assert not at.session_state["authenticated"]
    
    # Simulate clicking the Dev Bypass tab and the login button
    # The login button is "🚀 Sign In (Bypass)"
    # We find it in the buttons list and click it
    bypass_button = None
    for btn in at.button:
        if btn.label == "🚀 Sign In (Bypass)":
            bypass_button = btn
            break
            
    assert bypass_button is not None, "Could not find the bypass login button"
    bypass_button.click().run()
    
    # Verify the session state updated correctly
    assert at.session_state["authenticated"] is True
    assert at.session_state["user_email"] == "admin@marketcomps.dev"
    
    # Verify the "Log Out" button appears after login
    logout_found = any(btn.label == "🚪 Log Out" for btn in at.button)
    assert logout_found, "Log Out button should be visible after authentication"
