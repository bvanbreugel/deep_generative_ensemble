import pandas as pd 

def load_covid(relative_path = 'data/covid_data.csv', reduce_to=None):
    # from https://www.kaggle.com/code/imzeepo/covid-19-logistic-regression-random-forest
    
    try:
        covid = pd.read_csv(relative_path)
    except FileNotFoundError:
        raise FileNotFoundError('Could not find Covid Data.csv in data folder. Download from https://www.kaggle.com/datasets/meirnizri/covid19-dataset')
    
    # * sex: 1 for female and 2 for male.
    # * age: of the patient.
    # * classification: covid test findings. Values 1-3 mean that the patient was diagnosed with covid in different
    # * degrees. 4 or higher means that the patient is not a carrier of covid or that the test is inconclusive.
    # * patient type: type of care the patient received in the unit. 1 for returned home and 2 for hospitalization.
    # * pneumonia: whether the patient already have air sacs inflammation or not.
    # * pregnancy: whether the patient is pregnant or not.
    # * diabetes: whether the patient has diabetes or not.
    # * copd: Indicates whether the patient has Chronic obstructive pulmonary disease or not.
    # * asthma: whether the patient has asthma or not.
    # * inmsupr: whether the patient is immunosuppressed or not.
    # * hypertension: whether the patient has hypertension or not.
    # * cardiovascular: whether the patient has heart or blood vessels related disease.
    # * renal chronic: whether the patient has chronic renal disease or not.
    # * other disease: whether the patient has other disease or not.
    # * obesity: whether the patient is obese or not.
    # * tobacco: whether the patient is a tobacco user.
    # * usmr: Indicates whether the patient treated medical units of the first, second or third level.
    # * medical unit: type of institution of the National Health System that provided the care.
    # * intubed: whether the patient was connected to the ventilator.
    # * icu: Indicates whether the patient had been admitted to an Intensive Care Unit.
    # * date died: If the patient died indicate the date of death, and 9999-99-99 otherwise.

    # ### Data Preprocessing

    # Get rid of missing value samples
    # * except for INTUBED, PREGNANT, ICU columns since they have too many, delete these columns.

    cols = ['PNEUMONIA','DIABETES', 'COPD', 'ASTHMA', 'INMSUPR','HIPERTENSION', 
            'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY','RENAL_CHRONIC', 'TOBACCO']

    for col in cols :
        covid = covid[(covid[col] == 1)|(covid[col] == 2)]

    covid['target'] = [2 if row=='9999-99-99' else 1 for row in covid['DATE_DIED']]
    
    covid.drop(columns=['INTUBED','ICU','DATE_DIED'],inplace=True)

    # 'DATE_DIED' column to binary 'target' column
    
    if reduce_to is not None:
        covid = covid.sample(n=reduce_to, replace=False, random_state=42)
    y = covid['target']
    x = covid.drop('target', axis=1)

    return x, y

if __name__=='__main__':
    X, y = load_covid()