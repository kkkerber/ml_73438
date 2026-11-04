"""Adult schema: lab ranges and the UCI data dictionary.

Reference: https://archive.ics.uci.edu/dataset/2/adult
The literal '?' is an allowed missing-value marker, not a real class.
"""

NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]
CATEGORIES = {
    "workclass": "Private,Self-emp-not-inc,Self-emp-inc,Federal-gov,Local-gov,State-gov,Without-pay,Never-worked",
    "education": "Bachelors,Some-college,11th,HS-grad,Prof-school,Assoc-acdm,Assoc-voc,9th,7th-8th,12th,Masters,1st-4th,10th,Doctorate,5th-6th,Preschool",
    "marital-status": "Married-civ-spouse,Divorced,Never-married,Separated,Widowed,Married-spouse-absent,Married-AF-spouse",
    "occupation": "Tech-support,Craft-repair,Other-service,Sales,Exec-managerial,Prof-specialty,Handlers-cleaners,Machine-op-inspct,Adm-clerical,Farming-fishing,Transport-moving,Priv-house-serv,Protective-serv,Armed-Forces",
    "relationship": "Wife,Own-child,Husband,Not-in-family,Other-relative,Unmarried",
    "race": "White,Asian-Pac-Islander,Amer-Indian-Eskimo,Other,Black",
    "sex": "Female,Male",
    "native-country": "United-States,Cambodia,England,Puerto-Rico,Canada,Germany,Outlying-US(Guam-USVI-etc),India,Japan,Greece,South,China,Cuba,Iran,Honduras,Philippines,Italy,Poland,Jamaica,Vietnam,Mexico,Portugal,Ireland,France,Dominican-Republic,Laos,Ecuador,Taiwan,Haiti,Columbia,Hungary,Guatemala,Nicaragua,Scotland,Thailand,Yugoslavia,El-Salvador,Trinadad&Tobago,Peru,Hong,Holand-Netherlands",
    "income": "<=50K,>50K",
}
ALLOWED_CATEGORIES = {
    column: set(values.split(",")) | {"?"} for column, values in CATEGORIES.items()
}
EXPECTED_TYPES = {
    "age": "int64", "workclass": "category", "fnlwgt": "int64",
    "education": "category", "education-num": "int64", "marital-status": "category",
    "occupation": "category", "relationship": "category", "race": "category",
    "sex": "category", "capital-gain": "int64", "capital-loss": "int64",
    "hours-per-week": "int64", "native-country": "category", "income": "category",
}
RANGES = {
    "age": (17, 90), "hours-per-week": (1, 99),
    "capital-gain": (0, 99999), "capital-loss": (0, 4356),
}
