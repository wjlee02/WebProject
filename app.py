from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
import pandas as pd
import bcrypt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import joblib
from bson import ObjectId
from flask_socketio import SocketIO, send
from werkzeug.utils import secure_filename
import os


kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('min_max_scaler.pkl')
pca = joblib.load('pca_transformer.pkl')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

client = MongoClient('mongodb://localhost:27017/')
db = client['web_project']
user_col = db['user_profile']
user_basic_col = db['user']
other_col = db['web_project']

socketio = SocketIO(app)

UPLOAD_FOLDER = 'static/img/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return redirect(url_for('index'))

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/service')
def servide():
    return render_template('service.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        password = request.form['password']
        age = request.form['age']
        sex = request.form['sex']
        orientation = request.form['orientation']
        body_type = request.form['body_type']
        diet = request.form['diet']
        drinks = request.form['drinks']
        drugs = request.form['drugs']
        education = request.form['education']
        ethnicity = request.form['ethnicity']
        height = request.form['height']
        job = request.form['job']
        pets = request.form['pets']
        religion = request.form['religion']
        smokes = request.form['smokes']

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        if 'profile_image' in request.files:
            profile_image = request.files['profile_image']
            if profile_image.filename != '':
                image_filename = secure_filename(profile_image.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                profile_image.save(image_path)
                image_url = os.path.join('uploads', image_filename)  
            else:
                image_url = "/default_profile_image.jpg"  
        else:
            image_url = "/default_profile_image.jpg"  

        user_data = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'age': age,
            'sex': sex,
            'orientation': orientation,
            'body_type': body_type,
            'diet': diet,
            'drinks': drinks,
            'drugs': drugs,
            'education': education,
            'ethnicity': ethnicity,
            'height': height,
            'job': job,
            'pets': pets,
            'religion': religion,
            'smokes': smokes,
            'image_url': image_url  
        }

        result = user_col.insert_one(user_data)

        user_basic_data = {
            '_id': result.inserted_id,
            'email': email,
            'password': hashed_password
        }

        user_basic_col.insert_one(user_basic_data)

        flash('Registration successful!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')

        user = user_basic_col.find_one({'email': email})

        if user and bcrypt.checkpw(password, user['password']):
            session['user_email'] = user['email']
            # flash('Login successful!', 'success')
            return redirect(url_for('profile'))
        else:
            flash('Invalid User', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/profile')
def profile():
    if 'user_email' in session:
        user_email = session['user_email']
        user_profile = user_col.find_one({'email': user_email}, {"_id": 0, "password": 0, "email": 0})
        if not user_profile:
            flash('User profile not found.', 'error')
            return redirect(url_for('login'))
        return render_template('profile.html', user_profile=user_profile)
    else:
        flash('You need to login first.', 'warning')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


@app.route('/information')
def information():
    user_email = session.get('user_email')
    if not user_email:
        return redirect(url_for('login'))

    user_profile = user_col.find_one({'email': user_email}, {"_id": 0, "password": 0, "email": 0})

    if not user_profile:
        flash('User profile not found.', 'error')
        return redirect(url_for('login'))

    df = pd.DataFrame([user_profile])

    all_profiles = list(other_col.find({}, {"_id": 0, "password": 0, "email": 0}))
    df_all = pd.DataFrame(all_profiles)
    df_all.drop(['sign', 'membership'], axis=1, inplace=True)

    combined_df = pd.concat([df, df_all], ignore_index=True)
    combined_df.drop('name', axis=1, inplace=True)
    encoded_profiles = combined_df.copy()

    columns_to_encode = ['sex', 'orientation', 'body_type',
                         'diet', 'drinks', 'drugs', 'education', 'ethnicity',
                         'job', 'pets', 'religion', 'smokes']

    for column in columns_to_encode:
        encoded_cols = pd.get_dummies(combined_df[column], prefix=column, drop_first=True)
        encoded_profiles = pd.concat([encoded_profiles, encoded_cols], axis=1)
        encoded_profiles.drop(column, axis=1, inplace=True)

    col_to_scale = ['age', 'height']
    X = encoded_profiles[col_to_scale]

    X_scaled = scaler.transform(X)
    encoded_profiles[col_to_scale] = X_scaled

    if 'sex_m' in encoded_profiles.columns:
        encoded_profiles.drop(['sex_m'], axis=1, inplace=True)
    if 'image_url' in encoded_profiles.columns:
        encoded_profiles.drop(['image_url'], axis=1, inplace=True)

    encoded_profiles.dropna(inplace=True)

    feature_order = pca.feature_names_in_
    encoded_profiles = encoded_profiles.reindex(columns=feature_order, fill_value=0)

    X_pca = pca.transform(encoded_profiles)

    combined_df['membership'] = kmeans_model.predict(X_pca)

    user_cluster = combined_df.loc[0, 'membership']

    user_age = int(user_profile['age'])
    age_min = user_age - 5
    age_max = user_age + 5

    combined_df['age'] = combined_df['age'].astype(int)

    if user_profile['orientation'] == 'straight':
        if user_profile['sex'] == 'male':
            similar_profiles = combined_df[(combined_df['membership'] == user_cluster) &
                                           (combined_df['sex'] == 'f') &
                                           (combined_df['age'] >= age_min) &
                                           (combined_df['age'] <= age_max)].head(10).to_dict(orient='records')
        elif user_profile['sex'] == 'female':
            similar_profiles = combined_df[(combined_df['membership'] == user_cluster) &
                                           (combined_df['sex'] == 'm') &
                                           (combined_df['age'] >= age_min) &
                                           (combined_df['age'] <= age_max)].head(10).to_dict(orient='records')
    elif user_profile['orientation'] == 'gay':
        if user_profile['sex'] == 'male':
            similar_profiles = combined_df[(combined_df['membership'] == user_cluster) &
                                           (combined_df['sex'] == 'm') &
                                           (combined_df['age'] >= age_min) &
                                           (combined_df['age'] <= age_max)].head(10).to_dict(orient='records')
        elif user_profile['sex'] == 'female':
            similar_profiles = combined_df[(combined_df['membership'] == user_cluster) &
                                           (combined_df['sex'] == 'f') &
                                           (combined_df['age'] >= age_min) &
                                           (combined_df['age'] <= age_max)].head(10).to_dict(orient='records')
    elif user_profile['orientation'] == 'bisexual':
        similar_profiles = combined_df[(combined_df['membership'] == user_cluster) &
                                       (combined_df['age'] >= age_min) &
                                       (combined_df['age'] <= age_max)].head(10).to_dict(orient='records')
    else:
        flash('Invalid orientation.', 'error')
        return redirect(url_for('login'))

    return render_template('information.html', profiles=similar_profiles, user_profile=user_profile)




@app.route('/chat')
def chat_with_user():
    return render_template('chat.html')


if __name__ == '__main__':
    app.run(debug=True)
