"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
<<<<<<< Updated upstream
=======
from streamlit_option_menu import option_menu
>>>>>>> Stashed changes

# Data handling dependencies
import pandas as pd
import numpy as np
<<<<<<< Updated upstream
=======
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
import cufflinks as cf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from datetime import datetime
import plotly.graph_objects as go
import re
import altair as alt
import os
import streamlit as st
from PIL import Image
import base64

>>>>>>> Stashed changes

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

<<<<<<< Updated upstream
=======


# Data ALTERATIONS 


best_director_list = ['Quentin Tarantino', 'Michael Crichton', 'J.R.R. Tolkien', 'Lilly Wachowski', 'Stephen King', 'Ethan Coen', 'James Cameron', 'Luc Besson'
                        , 'Jonathan Nolan', 'Thomas Harris']
number_of_ratings_list = [109919, 65157, 62963, 60988, 59903, 51185, 51178, 44015, 42645, 36425]
title_cast_index_list = ['Samuel L. Jackson', 'Steve Buscemi', 'Keith David', 'Willem Dafoe', 'Robert De Niro', 'Brian Cox', 'Christopher Walken', 'Gérard Depardieu'
                         , 'Bruce Willis', 'Danny Glover', 'Morgan Freeman', 'Peter Stormare', 'Alec Baldwin', 'Nicolas Cage', 'Stanley Tucci', 'Julianne Moore'
                         , 'Richard Jenkins', 'Susan Sarandon', 'Stellan Skarsgård', 'John Goodman', 'Woody Harrelson', 'Tom Wilkinson', 'Antonio Banderas'
                         ,'Christopher McDonald', 'Val Kilmer', 'Jeff Bennett', 'Johnny Depp', 'Ed Harris', 'Donald Sutherland', 'John Leguizamo', 'Forest Whitaker'
                         , 'Harvey Keitel', 'John Cusack', 'Ray Liotta', 'Paul Giamatti', 'Luis Guzmán', 'Stephen Tobolowsky', 'George W. Bush', 'David Strathairn'
                         , 'Danny Trejo', 'Jim Broadbent', 'John Malkovich', 'Richard Riehle', 'Ewan McGregor', 'Kathy Bates', 'Robert Downey Jr.', 'Ving Rhames'
                         , 'Patricia Clarkson', 'Jim Cummings', 'William H. Macy', 'Billy Bob Thornton', 'Michael Gambon', 'Eric Roberts', 'Ben Kingsley'
                         , 'Robin Williams', 'William Hurt', 'Ron Perlman', 'Frank Welker', 'John Hurt', 'John Turturro']
title_cast_series_list = [83, 68, 61, 59, 58, 57, 57, 57, 56, 56, 56, 55, 55, 55, 54, 54, 54, 54, 53, 53, 53, 53, 52, 52, 52, 51, 50, 50, 49, 49, 49, 49, 49, 49, 48
                          , 48, 48, 47, 47, 47, 47, 47, 47, 47, 46, 46, 46, 46, 46, 46, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45]
genres_index_list = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Documentary', 'Crime', '(no genres listed)', 'Adventure', 'Sci-Fi', 'Children'
                     , 'Animation', 'Mystery', 'Fantasy', 'War', 'Western', 'Musical', 'Film-Noir', 'IMAX']
genres_series_list = [25606, 16870, 8654, 7719, 7348, 5989, 5605, 5319, 5062, 4145, 3595, 2935, 2929, 2925, 2731, 1874, 1399, 1054, 353, 195]
movie_year_series_list = [2513, 2488, 2406, 2374, 2173, 2034, 1978, 1838, 1724, 1691, 1632, 1498, 1446, 1255, 1172, 1028, 1024, 994, 971, 929]
movie_year_index_list = ['2015', '2016', '2014', '2017', '2013', '2018', '2012', '2011', '2009', '2010', '2008', '2007', '2006', '2005', '2004', '2003', '2002', '2019', '2001', '2000']
director_series_list = [28, 26, 26, 24, 19, 17, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12, 12, 11, 11]
director_index_list = ['See full summary', 'Woody Allen', 'Luc Besson', 'Stephen King', 'William Shakespeare', 'Ki-duk Kim', 'Lars von Trier', 'Tyler Perry'
                       , 'Alex Gibney', 'Robert Rodriguez', 'Takeshi Kitano', 'Peter Farrelly', 'David Mamet', 'Kevin Smith', 'Olivier Assayas', 'Sang-soo Hong'
                       , 'Clive Barker', 'Charles Band', 'John Sayles', 'Akira Toriyama']
rating_series_list = [2652977, 1959759, 1445230, 1270642, 880516, 656821, 505578, 311213, 159731, 157571]
rating_index_list = [4.0, 3.0, 5.0, 3.5, 4.5, 2.0, 2.5, 1.0, 1.5, 0.5]






#-------------------------------------------
#Background configs

logo = Image.open('resources/imgs/mmlogo.png')
st.set_page_config(page_title='Movie Muse', page_icon=logo)



@st.cache_data
def get_img_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    return base64.b64encode(img_data).decode()

img = get_img_as_base64('resources/imgs/bkr.png')
img1 = get_img_as_base64('resources/imgs/logom.png')

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: scroll;
color: black;
display : flex;
justify-content: centre;
align-items: centre;
}}

[data-testid="stSidebar"] {{
background-image: url("data:image/png;base64,{img1}");
background-size: fit;
background-position: center;
background-repeat: no-repeat;
}}
</style>"""
st.markdown(page_bg_img, unsafe_allow_html=True)


#made with streamlit info
hide_st_style = """
            <style>
            #MainMenu {visibility: show;}
            footer {visibility: hidden;}
            
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

>>>>>>> Stashed changes
# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
<<<<<<< Updated upstream
    page_options = ["Recommender System","Solution Overview","Business Proposal", "About Us" ,"Contact Us"]
=======
    page_options = ["Recommender System","Solution Overview", "Analysis","Business Proposal","About us", "Contact Us","Terms and Conditions"]
>>>>>>> Stashed changes

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
<<<<<<< Updated upstream
    page_selection = st.sidebar.selectbox("Menu", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### Videoligy')
=======
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Muse')
        st.write('### Movie Recommender System')
>>>>>>> Stashed changes
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
<<<<<<< Updated upstream
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.image('resources/imgs/first.jpg', use_column_width= True)
=======
    #Solution overview page
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.image('resources/imgs/first.png', use_column_width= True)
        st.image('resources/imgs/second.png', use_column_width= True)
>>>>>>> Stashed changes
        st.markdown(open('resources/About_solution.md').read())
        st.image('resources/imgs/third.jpg', use_column_width= True)
        st.markdown(open('resources/About2.md').read())
        st.image('resources/imgs/fourth.png', use_column_width= True)
        st.markdown(open('resources/About3.md').read())
        st.image('resources/imgs/fifth.png', use_column_width= True)
        st.markdown(open('resources/About4.md').read())
        st.markdown(open('resources/examples.md').read())
        st.image('resources/imgs/netflix.jpg', use_column_width= True)
        st.markdown(open('resources/benefits.md').read())

<<<<<<< Updated upstream
    if page_selection == "Business Proposal":
        st.title("Why our Product")
        st.markdown(open("resources/Pages/Business.md").read())


    if page_selection == "Contact Us":

        #Adding sidebar image
        #image = Image.open("resources/LOGOFINAL.png")
=======

    if page_selection == "Business Proposal":
        st.title("Why us")
        st.markdown(open('resources/Pages/Business.md').read())
        st.image('resources/imgs/still-life.jpg', use_column_width =True)
        st.markdown(open('resources/Pages/Business1.md').read())
        st.markdown(open('resources/Pages/Business2.md').read())



    # You may want to add more sections here for aspects such as an EDA,
    # or to  your business pitch.
    #---------------------------------------------------------------------

    if page_selection == "Analysis":
        st.title("Exploratory data analysis")
        st.write("Let's go through this insighful information that we gathered throughout the building process of our amazing app")


        selected = option_menu(
            menu_title = None,
            options = ["Explored Data", "General Movie titles Overview"],
            icons = ["database-fill-check","film"],#https://icons.getbootstrap.com/
            orientation = "horizontal"
        )

        if selected == "Explored Data":
            st.title("Explored Data visuals")
            
            if st.checkbox("Top 10 Users by Number of ratings"):
                st.title("The top 10 Users by Number of ratings")                 
                st.image('resources/Data visuals/tuser.png', use_column_width= True)
                    

            if st.checkbox("Distribution  of ratings"):
                st.title("The Distribution  of ratings")  
                st.image('resources/Data visuals/distro.png', use_column_width= True) 


            if st.checkbox("Mean ratings by number of ratings"):
                st.title("The mean ratings by number of ratings")
                st.image('resources/Data visuals/mratings.png', use_column_width= True)


            if st.checkbox("Top 5 of highest rated titles"):
                st.title("The top 5 of highest rated titles") 
                st.image('resources/Data visuals/fbest.png', use_column_width= True)



            if st.checkbox("Top 5 of worst rated titles"):
                st.title("The top 5 of worst rated titles")    
                st.image('resources/Data visuals/fworst.png', use_column_width= True)

            if st.checkbox("Number of users per rating"):
                st.title("Number of users per rating")
                st.image('resources/Data visuals/userRate.png', use_column_width= True)


            if st.checkbox("Numbers of movies per genre"):
                st.title("The numbers of movies per genre")
                st.image('resources/Data visuals/Pgenre.png', use_column_width= True)


            if st.checkbox("Number of movies per director"):
                st.title("Top Directors and Their Movie Counts")                
                st.image('resources/Data visuals/mdirect.png', use_column_width= True)


            if st.checkbox("Comparison of RMSE values between models"):
                st.title("Comparison of RMSE values between models")
                st.image('resources/Data visuals/rmse.png', use_column_width= True)

                

                


        if selected == "General Movie titles Overview":
            st.title("More insightful information")


            best_director_dict = {}
            for i in range(len(best_director_list)):
                    best_director_dict[best_director_list[i]] = number_of_ratings_list[i]
            if st.checkbox("Show Best Director Visual"):
                    
                    best_director_list2 = st.multiselect("Which directors would you like to see?", best_director_list)
                    best_director_emp_list = []
                    if len(best_director_list2) == 0:
                        st.subheader("Best Director Visual")
                        
                        source = pd.DataFrame({
                            'Director rating counts': number_of_ratings_list,
                            'Directors': best_director_list
                            })
                        

                        bar_chart = alt.Chart(source).mark_bar().encode(
                            y="Director rating counts:Q",
                            x='Directors:O',
                            
                        )
                        st.altair_chart(bar_chart, use_container_width=True)

                        
                            
                    else:
                        for i in best_director_list2:
                            best_director_emp_list.append(best_director_dict[i])
                        
                        st.subheader("Best Director Visual")
                        
                        source = pd.DataFrame({
                        'Director rating counts': best_director_emp_list, 'Directors': best_director_list2 })

                        bar_chart = alt.Chart(source).mark_bar().encode(
                                y="Director rating counts:Q",
                            x='Directors:O',
                        )
                        st.altair_chart(bar_chart, use_container_width=True)

            title_cast_dict = {}
            for i in range(len(title_cast_index_list)):
                    title_cast_dict[title_cast_index_list[i]] = title_cast_series_list[i]
            if st.checkbox("Show Title cast Visual"):
                    
                    title_cast_list2 = st.multiselect("Which cast would you like to see?", title_cast_index_list)
                    title_cast_emp_list = []
                    if len(title_cast_list2) == 0:
                        st.subheader("Title cast Visual")
                        
                        source = pd.DataFrame({
                            'Title cast count': title_cast_series_list,
                            'Title cast': title_cast_index_list
                            })
                        

                        bar_chart = alt.Chart(source).mark_bar().encode(
                            y="Title cast count:Q",
                            x='Title cast:O',
                            
                        )
                        st.altair_chart(bar_chart, use_container_width=True)

                        
                        

                            
                    else:
                        for i in title_cast_list2:
                            title_cast_emp_list.append(title_cast_dict[i])
                        
                        st.subheader("Title cast Visual")
                        
                        source = pd.DataFrame({
                        'Title cast count': title_cast_emp_list, 'Title cast': title_cast_list2 })

                        bar_chart = alt.Chart(source).mark_bar().encode(
                                y="Title cast count:Q",
                            x='Title cast:O',
                        )
                        st.altair_chart(bar_chart, use_container_width=True)

            rating_dict = {}
            for i in range(len(rating_index_list)):
                    rating_dict[rating_index_list[i]] = rating_series_list[i]
            if st.checkbox("Show Ratings Visual"):
                    rating_list2 = st.multiselect("Which rating would you like to see?", rating_index_list)
                    rating_emp_list = []
                    if len(rating_list2) == 0:
                        st.subheader("Ratings Visual")
                        
                        source = pd.DataFrame({
                            'Number of ratings': rating_series_list,
                            'Ratings': rating_index_list
                            })
                        

                        bar_chart = alt.Chart(source).mark_bar().encode(
                            y="Number of ratings:Q",
                            x='Ratings:O',
                            
                        )
                        st.altair_chart(bar_chart, use_container_width=True)
                        
                        
                            
                    else:
                        for i in rating_list2:
                            rating_emp_list.append(rating_dict[i])
                        
                        st.subheader("Ratings Visual")
                        
                        source = pd.DataFrame({
                        'Number of ratings': rating_emp_list, 'Ratings': rating_list2 })

                        bar_chart = alt.Chart(source).mark_bar().encode(
                                y="Number of ratings:Q",
                            x='Ratings:O',
                        )
                        st.altair_chart(bar_chart, use_container_width=True)

            director_dict = {}
            for i in range(len(director_index_list)):
                    director_dict[director_index_list[i]] = director_series_list[i]
            if st.checkbox("Show Directors Visual"):
                    director_list2 = st.multiselect("Which director would you like to see?", director_index_list)
                    director_emp_list = []
                    if len(director_list2) == 0:
                            st.subheader("Directors Visual")
                            source = pd.DataFrame({
                                'Director movies count': director_series_list,
                                'Directors': director_index_list
                                })
                            

                            bar_chart = alt.Chart(source).mark_bar().encode(
                                y="Director movies count:Q",
                                x='Directors:O'
                            )
                            st.altair_chart(bar_chart, use_container_width=True)
                            
                            
                    else:
                        for i in director_list2:
                            director_emp_list.append(director_dict[i])
                        
                        st.subheader("Directors Visual")
                        
                        source = pd.DataFrame({
                        'Director movies count': director_emp_list, 'Directors': director_list2 })

                        bar_chart = alt.Chart(source).mark_bar().encode(
                                y="Director movies count:Q",
                            x='Directors:O'
                        )
                        st.altair_chart(bar_chart, use_container_width=True)
            

            genre_dict = {}
            for i in range(len(genres_index_list)):
                    genre_dict[genres_index_list[i]] = genres_series_list[i]
            if st.checkbox("Show Top 20 genre Visual"):
                    genre_list2 = st.multiselect("Which genre would you like to see?", genres_index_list)
                    genre_emp_list = []
                    if len(genre_list2) == 0:
                        st.subheader("Top 20 genre Visual")
                        source = pd.DataFrame({
                            'Genre count': genres_series_list,
                            'Genre': genres_index_list
                            })
                        

                        bar_chart = alt.Chart(source).mark_bar().encode(
                            y='Genre count:Q',
                            x="Genre:O"
                        )
                        st.altair_chart(bar_chart, use_container_width=True)
                        
                    else:
                        for i in genre_list2:
                            genre_emp_list.append(genre_dict[i])
                        
                        st.subheader("Top 20 genre Visual")
                        
                        source = pd.DataFrame({
                        'Genre count': genre_emp_list, 'Genre': genre_list2 })

                        bar_chart = alt.Chart(source).mark_bar().encode(
                                y='Genre count:Q',
                            x="Genre:O"
                        )
                        st.altair_chart(bar_chart, use_container_width=True)

                            
            movie_year_dict = {}
            for i in range(len(movie_year_index_list)):
                    movie_year_dict[movie_year_index_list[i]] = movie_year_series_list[i]
            if st.checkbox("Show Top 20 movies by years Visual "):
                    movie_year_list2 = st.multiselect("Which year would you like to see?", movie_year_index_list)
                    movie_year_emp_list = []
                    if len(movie_year_list2) == 0:
                        st.subheader("Top 20 movies By years Visual")
                        source = pd.DataFrame({
                            'Number of movies': movie_year_series_list,
                            'Year': movie_year_index_list
                            })
                        

                        bar_chart = alt.Chart(source).mark_bar().encode(
                            y='Number of movies:Q',
                            x="Year:O"
                        ).properties(width=500,height=500)
                        st.altair_chart(bar_chart, use_container_width=True)
                        
                                
                    else:
                            for i in movie_year_list2:
                                movie_year_emp_list.append(movie_year_dict[i])
                            
                            st.subheader("Top 20 movies by years Visual")
                            
                            source = pd.DataFrame({
                            'Number of movies': movie_year_emp_list, 'Year': movie_year_list2 })

                            bar_chart = alt.Chart(source).mark_bar().encode(
                                    y='Number of movies:Q',
                                x="Year:O"
                            )
                            st.altair_chart(bar_chart, use_container_width=True)
        
                              
    #-----------------------------------------------------

    if page_selection == "About us":
        st.title("About Us")

        selected = option_menu(
            menu_title = None,
            options = ["Team","About the App"],
            icons = ["person","app-indicator"],#https://icons.getbootstrap.com/
            orientation = "horizontal"
            )




        #--------------------------------------------------------------------------------


        if selected == "Team": 
            st.subheader("Meet Our Team")



            def display_team_member(image_path, member_name, contact_info, role):
                st.image(image_path, caption=member_name, width=200)
                st.write(f"**Name:** {member_name}")
                st.write(f"**Contact Information:** {contact_info}")
                st.write(f"**Role:** {role}")
                st.write("---") 

            def about_us():
                st.title("About Us")

            
            path_to_images_folder = ('resources/imgs')

            team_members = [
                {'image_file': 'Thembelani.jpg', 'name': 'Thembela Ndlovu', 'contact_info': 'tehmbela3@gmail.com', 'role': 'Project Manager'},
                {'image_file': 'wandile.jpg', 'name': 'Wandile Hadebe', 'contact_info': 'wandilehadebe@gmail.com', 'role': 'Data Scientist'},
                {'image_file': 'Arnold.jpg', 'name': 'Arnold Ripfumelo Mathiane', 'contact_info': 'arnoldmatiane@gmail.com', 'role': 'Data Engineer'},
                {'image_file': 'Abi.jpg', 'name': 'Abidence Bvindi', 'contact_info': 'abvindi@gmail.com', 'role': 'Data Analyst'},
                {'image_file': 'Nnete.jpg', 'name': 'Nnete Mogane', 'contact_info': 'moganennete08@gmail.com', 'role': 'Research Scientist'},
                {'image_file': 'Kwanele.jpg', 'name': 'Sandile Khumalo', 'contact_info': 'sandilemtungwa16@gmail.com', 'role': 'Finance Manager'},
                {'image_file': 'neo.png', 'name': 'Neo Nenzhelele', 'contact_info': 'nenzneo@gmail.com', 'role': 'Software Developer'}
            ]

            for member in team_members:
                image_path= os.path.join(path_to_images_folder, member['image_file'])
                display_team_member(image_path, member['name'], member['contact_info'], member['role'])


        if selected == "About the App": 
            st.subheader("Movie Muse")
            st.markdown(open('resources/Pages/about.md').read())

#-------------------------------------------------------------------------------------------------------

    if page_selection == "Contact Us":

>>>>>>> Stashed changes
        
        with st.form("form1", clear_on_submit=True):
            st.subheader("Get in touch with us")
            name = st.text_input("Enter full name")
            email = st.text_input("Enter email")
            message = st.text_area("Message")
            
            submit = st.form_submit_button("Submit Form")
            if submit:
                st.write("Your form has been submitted and we will be in touch 🙂")
<<<<<<< Updated upstream
            


    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.unsplash.com/photo-1535303311164-664fc9ec6532?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1887&q=80");
    background-size: 100%;
    background-position: top;
    background-repeat: no-repeat;
    background-attachment: scroll;
    color: black;
    display : flex;
    justify-content: centre;
    align-items: centr;
    }}

    [data-testid="stHeader"].main
    </style>"""

    st.markdown(page_bg_img , unsafe_allow_html=True)
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    
    
=======

    #----------------------------------------------------------------------
    
    if page_selection == "Terms and Conditions":
        st.title("Terms and Conditions")
        st.title("Streamlit Terms and Conditions") 
        st.image('resources/imgs/slogo.png', use_column_width= True)
        st.markdown(open('resources/slitTerms.md', encoding='utf-8').read())


        
        


>>>>>>> Stashed changes
if __name__ == '__main__':
    main()
