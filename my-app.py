from flask import *  
import sqlite3

import numpy as np
import pandas as pd
# from gui_stuff import *

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze','continuous_sneezing','shivering','chills','fatigue','cough','headache']


disease=['Fungal infection',
'Allergy',
'GERD',
'Chronic cholestasis',
'Drug Reaction',
'Peptic ulcer diseae',
'AIDS',
'Diabetes',
'Gastroenteritis',
'Bronchial Asthma',
'Hypertension',
' Migraine',
'Cervical spondylosis',
'Paralysis (brain hemorrhage)',
'Jaundice',
'Malaria',
'Chicken pox',
'Dengue',
'Typhoid',
'hepatitis A',
'Hepatitis B',
'Hepatitis C',
'Hepatitis D',
'Hepatitis E',
'Alcoholic hepatitis',
'Tuberculosis',
'Common Cold',
'Pneumonia',
'Dimorphic hemmorhoids(piles)',
'Heart Diseases',
'Varicoseveins',
'Hypothyroidism',
'Hyperthyroidism',
'Hypoglycemia',
'Osteoarthristis',
'Arthritis',
'(vertigo) Paroymsal  Positional Vertigo',
'Acne',
'Urinary tract infection',
'Psoriasis',
'Impetigo']


urllinks = [ 'https://www.medicalnewstoday.com/articles/317970', 
'https://www.healthline.com/health/allergies#:~:text=An%20allergy%20is%20an%20immune,healthy%20by%20fighting%20harmful%20pathogens.',
'https://www.healthline.com/health/gerd', 
'https://www.healthline.com/health/cholestasis',
'https://www.healthline.com/health/drug-allergy#:~:text=A%20drug%20allergy%20is%20an,%2C%20fever%2C%20and%20trouble%20breathing.',
'https://www.mayoclinic.org/diseases-conditions/peptic-ulcer/symptoms-causes/syc-20354223',
'https://www.mayoclinic.org/diseases-conditions/hiv-aids/symptoms-causes/syc-20373524#:~:text=Acquired%20immunodeficiency%20syndrome%20(AIDS)%20is,to%20fight%20infection%20and%20disease.',
'https://www.healthline.com/health/diabetes#:~:text=Diabetes%20mellitus%2C%20commonly%20known%20as,the%20insulin%20it%20does%20make.',
'https://www.mayoclinic.org/diseases-conditions/viral-gastroenteritis/symptoms-causes/syc-20378847',
'https://www.healthline.com/health/asthma',
'https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/symptoms-causes/syc-20373410',
'https://www.healthline.com/health/migraine#:~:text=Migraine%20is%20a%20neurological%20condition,families%20and%20affect%20all%20ages.',
'https://www.healthline.com/health/cervical-spondylosis',
'https://www.healthline.com/health/lobar-intracerebral-hemorrhage#:~:text=Intracerebral%20hemorrhage%20(ICH)%20is%20when,one%20side%20of%20your%20body.',
'https://www.healthline.com/health/jaundice-yellow-skin',
'https://www.mayoclinic.org/diseases-conditions/malaria/symptoms-causes/syc-20351184',
'https://www.mayoclinic.org/diseases-conditions/chickenpox/symptoms-causes/syc-20351282',
'https://www.healthline.com/health/dengue-fever#TOC_TITLE_HDR_1',
'https://www.mayoclinic.org/diseases-conditions/typhoid-fever/symptoms-causes/syc-20378661',
'https://www.mayoclinic.org/diseases-conditions/hepatitis-a/symptoms-causes/syc-20367007',
'https://www.mayoclinic.org/diseases-conditions/hepatitis-b/symptoms-causes/syc-20366802',
'https://www.mayoclinic.org/diseases-conditions/hepatitis-c/symptoms-causes/syc-20354278#:~:text=Hepatitis%20C%20is%20a%20viral,HCV)%20spreads%20through%20contaminated%20blood.',
'https://www.healthline.com/health/delta-agent-hepatitis-d#:~:text=Hepatitis%20D%2C%20also%20known%20as,hepatitis%20D%20virus%20(HDV).',
'https://www.healthline.com/health/hepatitis-e',
'https://www.mayoclinic.org/diseases-conditions/alcoholic-hepatitis/symptoms-causes/syc-20351388',
'https://www.mayoclinic.org/diseases-conditions/tuberculosis/symptoms-causes/syc-20351250',
'https://www.mayoclinic.org/diseases-conditions/common-cold/symptoms-causes/syc-20351605',
'https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204#:~:text=Pneumonia%20is%20an%20infection%20that,and%20fungi%2C%20can%20cause%20pneumonia.',
'https://www.spectrumhealth.org/patient-care/digestive-health-and-disorders/colorectal/hemorrhoids',
'https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118',
'https://www.mayoclinic.org/diseases-conditions/varicose-veins/symptoms-causes/syc-20350643',
'https://www.medicinenet.com/hypothyroidism/article.htm',
'https://www.mayoclinic.org/diseases-conditions/hyperthyroidism/symptoms-causes/syc-20373659#:~:text=Hyperthyroidism%20(overactive%20thyroid)%20occurs%20when,a%20rapid%20or%20irregular%20heartbeat.',
'https://www.mayoclinic.org/diseases-conditions/hypoglycemia/symptoms-causes/syc-20373685#:~:text=Hypoglycemia%20is%20a%20condition%20in,who%20don\'t%20have%20diabetes',
'https://www.mayoclinic.org/diseases-conditions/arthritis/symptoms-causes/syc-20350772#:~:text=Arthritis%20is%20the%20swelling%20and,are%20osteoarthritis%20and%20rheumatoid%20arthritis.',
'https://www.arthritis.org/health-wellness/about-arthritis/understanding-arthritis/what-is-arthritis',
'https://www.uofmhealth.org/health-library/hw263714#:~:text=Benign%20paroxysmal%20positional%20vertigo%20(BPPV)%20causes%20a%20whirling%2C%20spinning,by%20rolling%20over%20in%20bed.',
'https://www.mayoclinic.org/diseases-conditions/acne/symptoms-causes/syc-20368047#:~:text=Acne%20is%20a%20skin%20condition,affects%20people%20of%20all%20ages.',
'https://www.mayoclinic.org/diseases-conditions/urinary-tract-infection/symptoms-causes/syc-20353447',
'https://www.healthline.com/health/psoriasis',
'https://www.healthline.com/health/impetigo' ]


















l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------


def NaiveBayes(sym1, sym2, sym3, sym4, sym5):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [sym1, sym2, sym3, sym4, sym5]
    print(psymptoms)
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        return disease[a]
        
    else:
        return None





# def Randomforest(symptoms):
#     from sklearn.ensemble import RandomForestClassifier
#     clf4 = RandomForestClassifier()
#     clf4 = clf4.fit(X, np.ravel(y))

#     # Calculating accuracy
#     from sklearn.metrics import accuracy_score
#     y_pred = clf4.predict(X_test)
#     print(accuracy_score(y_test, y_pred))
#     print(accuracy_score(y_test, y_pred, normalize=False))
#     # -----------------------------------------------------

#     psymptoms = symptoms

#     l2 = [0] * len(l1)
#     for k in range(0, len(l1)):
#         for z in psymptoms:
#             if z == l1[k]:
#                 l2[k] = 1

#     inputtest = [l2]
#     predict = clf4.predict(inputtest)
#     predicted = predict[0]

#     h = 'no'
#     for a in range(0, len(disease)):
#         if predicted == a:
#             h = 'yes'
#             break

#     if h == 'yes':
#         return disease[a]
#     else:
#         return None

def Randomforest(symptoms):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import numpy as np

    # Ensure X and y are defined globally or pass them as arguments
    global X, y, l1, disease

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4.fit(X_train, np.ravel(y_train))

    # Predict on the test set
    y_pred = clf4.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    print("Number of correct predictions:", accuracy_score(y_test, y_pred, normalize=False))
    
    # Transform the input symptoms into the model's input format
    psymptoms = symptoms
    l2 = [0] * len(l1)
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted = predict[0]

    # Find the corresponding disease
    if predicted in range(len(disease)):
        return disease[predicted]
    else:
        return None





def DecisionTree(sym1,sym2,sym3,sym4,sym5):

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [sym1,sym2,sym3,sym4,sym5]

    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        return disease[a]
        
    else:
        return None




def LogisticRegression(sym1,sym2,sym3,sym4,sym5):

    from sklearn import linear_model
    clf3 = linear_model.LogisticRegression()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [sym1,sym2,sym3,sym4,sym5]

    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        return disease[a]
        
    else:
        return None




def KMeans(sym1,sym2,sym3,sym4,sym5):

    from sklearn import cluster
    clf3 = cluster.KMeans()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [sym1,sym2,sym3,sym4,sym5]

    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        return disease[a]
        
    else:
        return None








app = Flask(__name__)  

DATABASE = 'userdatabase.db'


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
        

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


def insert_user(name, phone, password, gender):
    con = get_db()
    status = False
    try:
        cur = con.cursor()  
        cur.execute("INSERT INTO users(name,phone,pass,gender)VALUES(?,?,?,?)",(name, phone, password,gender))  
        con.commit()
        status = True
    except:  
        con.rollback()
        status = False
    finally:  
        return status
         
    

def checkLogin(phone, password):
    conn = get_db()
    user = query_db('select * from users where phone = ? AND pass = ?',
                (phone,password), one=True)
    return user
    

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/register')
def register():
    return render_template('register.html') 
    
@app.route('/registeruser',methods = ['POST'])
def registeruser():
    name=request.form['name']    
    phone=request.form['phone']  
    password=request.form['pass'] 
    gender=request.form['gender']
    result = insert_user(name,phone,password,gender)
    if result == True:
        return  "<script>alert('user registration successfull.'); window.open('/home','_self')</script>"
    else:
        return  "<script>alert('user registration Fail.'); window.open('/register','_self')</script>"


@app.route('/login',methods = ['POST'])
def login():
    phone=request.form['phone']  
    password=request.form['pass'] 
    user=checkLogin(phone, password)
    if user is None:
       return  "<script>alert('Phone or password did not match.'); window.open('/','_self')</script>"
    else:
        return render_template('home.html') 
        # "<script>alert('user registration successfull.'); window.open('/home','_self')</script>"
        
        
        
@app.route('/naivebayes')
def opt_naivebayes():
    return render_template('naivebayes.html')
    
    
@app.route('/randomforest')
def opt_randomforest():
    return render_template('randomforest.html')
    
@app.route('/kmeans')
def opt_kmeans():
    return render_template('kmeans.html')
    
@app.route('/logisticregression')
def opt_logisticregression():
    return render_template('logisticregression.html')
    
    
@app.route('/payment')
def payment():
    return render_template('payment.html')

#decision tree
@app.route('/decisiontree')
def decisiontree():
    return render_template('decisiontree.html')


@app.route('/home',methods=['GET'])
def home():
    return render_template('home.html')
    

      
@app.route('/naivebayes_req',methods=['POST'])
def naivebayes_req():
    sym1=request.form['sym1']  
    sym2=request.form['sym2'] 
    sym3=request.form['sym3'] 
    sym4=request.form['sym4'] 
    sym5=request.form['sym5']
    result = NaiveBayes(sym1,sym2,sym3,sym4,sym5)
    
    #print("index of dises :")
    #print(disease.index(result))
    url = urllinks[disease.index(result)]
    #print(urllinks[disease.index(result)])
    
    
    if result == None:
        return  "<script>alert('Not Found.'); window.open('/home','_self')</script>"
    else:
        return render_template('pridiction.html', m_result = result, m_link = url)
        


# @app.route('/randomforest_req',methods=['POST'])
# def randomforest_req():
#     # sym1=request.form['sym1']  
#     # sym2=request.form['sym2']
#     # sym3=request.form['sym3']
#     # sym4=request.form['sym4']
#     # sym5=request.form['sym5']
#     # result = Randomforest(sym1,sym2,sym3,sym4,sym5)
    
#     # #print("index of dises :")
#     # #print(disease.index(result))
#     # url = urllinks[disease.index(result)]
#     # #print(urllinks[disease.index(result)])
    
    
#     # if result == None:
#     #     return  "<script>alert('Not Found.'); window.open('/home','_self')</script>"
#     # else:
#     #     return render_template('pridiction.html', m_result = result, m_link = url)


#     symptoms = request.form.getlist('symptoms[]')

    
#     # Assuming Randomforest function can take a list of symptoms now
#     result = Randomforest(*symptoms[:5])  # Pass only the first 5 symptoms if more than 5 are selected
    
#     url = urllinks[disease.index(result)] if result in disease else None

#     if result is None:
#         return "<script>alert('Not Found.'); window.open('/home','_self')</script>"
#     else:
#         return render_template('pridiction.html', m_result=result, m_link=url)

@app.route('/randomforest_req', methods=['POST'])
def randomforest_req():
    symptoms = request.form.getlist('symptoms[]')
    # print('--------------------------')
    print(symptoms)
    
    if len(symptoms) < 3:
        return "<script>alert('Please select at least 3 symptoms.'); window.open('/home','_self')</script>"
    elif len(symptoms) > 5:
        return "<script>alert('You can select a maximum of 5 symptoms.'); window.open('/home','_self')</script>"
    
    result = Randomforest(symptoms[:5])  # Pass only the first 5 symptoms if more than 5 are selected
    
    url = urllinks[disease.index(result)] if result in disease else None

    if result is None:
        return "<script>alert('Not Found.'); window.open('/home','_self')</script>"
    else:
        return render_template('pridiction.html', m_result=result, m_link=url)
    
@app.route('/decision_tree_req',methods=['POST'])
def decision_tree_req():
    sym1=request.form['sym1']  
    sym2=request.form['sym2'] 
    sym3=request.form['sym3'] 
    sym4=request.form['sym4'] 
    sym5=request.form['sym5']
    result = DecisionTree(sym1,sym2,sym3,sym4,sym5)
    
    #print("index of dises :")
    #print(disease.index(result))
    url = urllinks[disease.index(result)]
    #print(urllinks[disease.index(result)])
    
    if result == None:
        return  "<script>alert('Not Found.'); window.open('/home','_self')</script>"
    else:
        return render_template('pridiction.html', m_result = result, m_link = url)
        
        
@app.route('/kmeans_req',methods=['POST'])
def kmeans_req():
    sym1=request.form['sym1']  
    sym2=request.form['sym2'] 
    sym3=request.form['sym3'] 
    sym4=request.form['sym4'] 
    sym5=request.form['sym5']
    result = KMeans(sym1,sym2,sym3,sym4,sym5)
    
    #print("index of dises :")
    #print(disease.index(result))
    url = urllinks[disease.index(result)]
    #print(urllinks[disease.index(result)])
    
    if result == None:
        return  "<script>alert('Not Found.'); window.open('/home','_self')</script>"
    else:
        return render_template('pridiction.html', m_result = result, m_link = url)
        
        
@app.route('/logisticregression_req',methods=['POST'])
def logisticregression_req():
    sym1=request.form['sym1']  
    sym2=request.form['sym2'] 
    sym3=request.form['sym3'] 
    sym4=request.form['sym4'] 
    sym5=request.form['sym5']
    result = LogisticRegression(sym1,sym2,sym3,sym4,sym5)
    
    #print("index of dises :")
    #print(disease.index(result))
    url = urllinks[disease.index(result)]
    #print(urllinks[disease.index(result)])
    
    if result == None:
        return  "<script>alert('Not Found.'); window.open('/home','_self')</script>"
    else:
        return render_template('pridiction.html', m_result = result, m_link = url)




    
    
    
   
if __name__ == '__main__':
   print(len(disease))
   print(len(urllinks))
   
   app.run(debug = True)  