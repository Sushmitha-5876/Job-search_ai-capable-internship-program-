"""Maintainable company options used only by the Streamlit UI."""

PRODUCT_AND_BIG_TECH = [
    "Adobe", "Airbnb", "Amazon", "Apple", "Atlassian", "Cisco", "Dell",
    "GitHub", "Google", "HP", "IBM", "Intel", "LinkedIn", "Meta",
    "Microsoft", "Netflix", "NVIDIA", "Oracle", "Qualcomm", "Salesforce",
    "SAP", "Uber", "VMware",
]

FINTECH_AND_BANKING = [
    "CRED", "Goldman Sachs", "Groww", "JPMorgan Chase", "Mastercard",
    "Morgan Stanley", "PayPal", "PhonePe", "Razorpay", "Visa", "Wells Fargo",
]

IT_SERVICES_AND_CONSULTING = [
    "Accenture", "Capgemini", "Cognizant", "Deloitte", "EY", "HCLTech",
    "Infosys", "KPMG", "LTIMindtree", "PwC", "TCS", "Tech Mahindra", "Wipro",
]

ECOMMERCE_AND_STARTUPS = [
    "Flipkart", "Freshworks", "Meesho", "Swiggy", "Walmart Global Tech",
    "Zomato", "Zoho",
]

COMPANIES = sorted(set(
    PRODUCT_AND_BIG_TECH
    + FINTECH_AND_BANKING
    + IT_SERVICES_AND_CONSULTING
    + ECOMMERCE_AND_STARTUPS
))

CUSTOM_COMPANY_OPTION = "Custom (enter company name)"
COMPANY_OPTIONS = COMPANIES + [CUSTOM_COMPANY_OPTION]
