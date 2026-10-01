"""
locations.py

Locations available in the job-search location dropdown.
"""

INDIAN_CITIES = [
    "Bengaluru",
    "Mumbai",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Delhi",
    "Delhi NCR",
    "Gurugram",
    "Noida",
    "Kolkata",
    "Ahmedabad",
    "Chandigarh",
    "Jaipur",
    "Kochi",

    "Mysuru",
    "Mangaluru",
    "Hubballi",
    "Dharwad",
    "Belagavi",
    "Shivamogga",

    "Coimbatore",
    "Madurai",
    "Tiruchirappalli",
    "Salem",
    "Tiruppur",
    "Vellore",
    "Chengalpattu",

    "Warangal",
    "Visakhapatnam",
    "Vijayawada",
    "Guntur",
    "Tirupati",

    "Thiruvananthapuram",
    "Kozhikode",
    "Thrissur",

    "Nagpur",
    "Nashik",
    "Aurangabad",
    "Kolhapur",
    "Navi Mumbai",

    "Surat",
    "Vadodara",
    "Rajkot",
    "Gandhinagar",

    "Udaipur",
    "Jodhpur",
    "Kota",

    "Indore",
    "Bhopal",

    "Lucknow",
    "Kanpur",
    "Agra",
    "Varanasi",
    "Prayagraj",

    "Patna",
    "Ranchi",
    "Jamshedpur",

    "Bhubaneswar",
    "Cuttack",

    "Durgapur",
    "Guwahati",

    "Mohali",
    "Ludhiana",
    "Amritsar",
    "Faridabad",

    "Raipur",
    "Goa",
]

NO_PREFERENCE_LOCATION_OPTION = "No preference (use resume location)"

CUSTOM_LOCATION_OPTION = "Custom (enter location)"

LOCATION_OPTIONS = (
    [NO_PREFERENCE_LOCATION_OPTION]
    + INDIAN_CITIES
    + [CUSTOM_LOCATION_OPTION]
)