import unittest
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

class FlaskAppHomeTest(unittest.TestCase):
    def setUp(self):
        # Initialize WebDriver options and service
        options = Options()
        options.headless = True  # Run in headless mode
        service = Service(executable_path='/usr/local/bin/geckodriver',
                          log_path='geckodriver.log',
                          verbose=True)  # Enable verbose logging for troubleshooting

        # Initialize the Firefox WebDriver with the specified service and options
        self.driver = webdriver.Firefox(service=service, options=options)
        self.driver.implicitly_wait(10)  # Wait up to 10 seconds for elements to become available

    def test_home_page_content(self):
        # Navigate to the specified route of your Flask application
        self.driver.get("http://localhost:5000/home")

        # Example assertion: Check for a specific element, text, or title on the page
        # Adjust the assertion as needed based on your application's content
        self.assertIn("Expected Text or Title", self.driver.title)

        # You can add more assertions here to check for specific elements or content
        # For example:
        # element = self.driver.find_element_by_id("element_id")
        # self.assertEqual("Expected Text", element.text)

    def tearDown(self):
        # Quit the WebDriver, closing the headless browser session
        self.driver.quit()

if __name__ == "__main__":
    unittest.main()
