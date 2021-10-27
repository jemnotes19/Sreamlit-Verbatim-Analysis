#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import re

import pandas as pd
import base64
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer


import plotly.express as px




import gensim
import gensim.corpora as corpora


import scipy.sparse

import spacy


import pyLDAvis
import pyLDAvis.gensim_models




def main():
    
    
    st.set_page_config(layout="wide")
    st.image('./Desktop/rx_new_logo.png')

    
    nav = st.sidebar.selectbox("Navigation", [ "Word Cloud", "Keyword Finder", "TextBlob Sentiment Analysis","Topic Modelling"])
    
    pos_tag_dict = {"Adjective":"ADJ","Adverb":"ADV", "Noun":"NOUN","Proper Noun":"PROPN", "Verb":"VERB"}
    def clean_data(df, col_name):
        # remove all rows that are blanks    
        series_clean = df[col_name].dropna()
        
        df_clean = series_clean.to_frame(col_name)
        
        # clean the data starting with lower case 
        df_clean[col_name] = df_clean[col_name].str.lower()
        
        # clean the data by removing all punctuation
        df_clean[col_name] = df_clean[col_name].str.replace(r'[^\w\s]+', '')
        
        # clean the data by removing numbers
        df_clean[col_name] = df_clean[col_name].str.replace(r'\d+', '')
        
        # clean data by removing quotation marks
        df_clean[col_name] = df_clean[col_name].str.replace('[‘’“”…]', '')
        
        # clean data by removing white space \n
        df_clean[col_name] = df_clean[col_name].str.replace('\n', '')
        
        return df_clean
        
    def lemmatization(texts, allowed_postags = ["NOUN","ADJ","VERB","ADV"]):
        nlp = spacy.load("en_core_web_sm", disable =["parser","ner"])
        texts_out = []
        for text in texts:
            doc = nlp(text)
            new_text =[]
            for token in doc:
                if token.pos_ in allowed_postags:
                    new_text.append(token.lemma_)
            final = " ".join(new_text)
            texts_out.append(final)
        return texts_out
    
    def convertdf_to_dataword(input_df, col_name):
        datalist = input_df[col_name].tolist()
        output = [re.sub(r"[^\w]", " ",  x).split() for x in datalist]
        return output
    
    def collect_bigrams(input_list):
        collection = {}
        for each in input_list:
            for every in each:
                if "_" in every:
                    if every not in collection:
                        collection[every] = [1]
                    else:
                        collection[every][0] +=1
        return collection
    
    def list_to_df(bigram_list, col_name):
        list_for_df = [" ".join(items) for items in bigram_list]
        output_df = pd.DataFrame (list_for_df, columns = [col_name])
        return output_df
    
    def topic_search(topic_df, data_list):
        topics_dict = {x:topic_df[x].dropna().str.lower().str.replace(r'[^\w\s]+', '').tolist() for x in topic_df.columns}
        result_dict ={x:[] for x in topic_df.columns}
        other_list = []
        for topics in topics_dict:
            for each_doc in data_list:
                for each_item in each_doc:
                    if each_item in topics_dict[topics]:
                        if each_doc not in result_dict[topics]:
                            result_dict[topics].append(" ".join(each_doc))
                        break
                        
        for docs in data_list:
            if " ".join(docs) not in result_dict.values():
                other_list.append(" ".join(docs))
    
        result_dict["Other"] = other_list
        return result_dict
    
    from sklearn.feature_extraction import text 
    def remove_stopwords(doc):
        removed = []
        for response in doc:
            temp_list = []
            for each_text in response:
                if each_text not in text.ENGLISH_STOP_WORDS:
                    temp_list.append(each_text)
            if not temp_list:
                pass
            else:
                removed.append(temp_list)
        return removed
    
    if nav == "TextBlob Sentiment Analysis":
        st.title('Text Blob Sentiment Analysis')
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.form(key='my_form'):
                data_sheet = st.file_uploader('File uploader', type = ['xlsx'])
                excel_sheet_name = st.text_input("Please enter sheet name here")
                col_name = st.text_input('Please enter column name')
                st.text("")
                posneg_interval = st.slider('Sentiment Interval(Neutral)', 
                                            value =[-1.0,1.0], 
                                            step = 0.1,
                                            help = "Define the Polarity interval for neutral range")
                submit_button = st.form_submit_button(label='Process Data')

        # Define the limits for the sentiments
        pos = posneg_interval[1]
        neg = posneg_interval[0]
        
        
        # setup the sentiment range function
        def sentiment_intervals(input):
            if input<neg:
                return "Negative"
            elif input>pos:
                return "Positive"
            else:
                return "Neutral"
               
        def process_data(input_df, col_name):   
            # Use textblob to apply polarity
            pol = lambda x: TextBlob(x).sentiment.polarity
            input_df["Polarity"] = input_df[col_name].apply(pol)
            input_df["Sentiment"] = input_df["Polarity"].apply(sentiment_intervals)
            
            return input_df
            
        
        #old method of dowload hpyerlink before streamlit added the download button function
        #def get_table_download_link_csv(df):
            #csv = df.to_csv(index=False)
            #csv = df.to_csv().encode()
            #b64 = base64.b64encode(csv.encode()).decode() 
            #b64 = base64.b64encode(csv).decode()
            #href = f'<a href="data:file/csv;base64,{b64}" download="Sentiment Analysis.csv" target="_blank">Download csv file</a>'
            #return href
        
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        
        def sentiment_analysis(df, col_name):       
            return process_data(clean_data(df, col_name), col_name)
        
        
        if data_sheet is not None:
            #file_data = { "file_type": data_sheet.type, 'format':data_sheet.__format__}
            #checks the file type of the uploaded file
            # Excel sheets are typed as "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            # csv sheets are typed as "application/vnd.ms-excel"
            #st.write(file_data)   
            
            if data_sheet.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df_raw = pd.read_excel(data_sheet, sheet_name = excel_sheet_name)  
                
               
            result = sentiment_analysis(df_raw, col_name)
            
            summary = result.groupby( ["Sentiment"] ).size().to_frame(name = 'count').reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                
                st.write(result)
                
                csv = convert_df(result)
                
                st.write("")
                st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name='Sentiment Analysis.csv',
                        mime='text/csv')
                #old hypelink code before streamlit added the download button    
                #st.markdown(get_table_download_link_csv(result), unsafe_allow_html=True)
            with col2:
                pie_chart = px.pie(data_frame = summary, 
                   values = "count", 
                   names = "Sentiment",
                   color = "Sentiment",
                   color_discrete_map = {"Positive": "green", "Neutral":"khaki", "Negative": "red"},
                   title = "Sentiment Summary",
                   template = "presentation")
                st.plotly_chart(pie_chart)
            
    
    if nav == "Word Cloud":
        st.title('Word Cloud')
        
        col1, col2 = st.columns([1,2])
        
        with col1:
            
            with st.form(key='my_form'):
                data_sheet = st.file_uploader('File uploader', type = ['xlsx'])
                excel_sheet_name = st.text_input("Please enter sheet name here")
                col_name = st.text_input('Please enter column name')
                topword_count = st.slider('Number of Top Words', min_value=1, max_value=30)
                lemma_on = st.checkbox('Apply Lemmatization')
                submit_button = st.form_submit_button(label='Process Data')

        if data_sheet is not None:
            if data_sheet.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df_raw = pd.read_excel(data_sheet, sheet_name = excel_sheet_name) 
            
            
            df_clean = clean_data(df_raw, col_name)
            
            if lemma_on:
                df_clean = df_clean.apply(lemmatization)
            
            st.subheader('Cleaned Responses')
            st.write(df_clean)
            
            cv = CountVectorizer(stop_words='english')
            data_cv = cv.fit_transform(df_clean[col_name])
            data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
            data_dtm.index = df_clean[col_name]
            
            top_dict = {}
            for c in data_dtm.columns:
                top_dict[c]= data_dtm[c].sum()
            
            topword_list = sorted(top_dict.items(), key=lambda x: x[1], reverse = True)[:topword_count]
            topword_dict = {k:v for (k,v) in topword_list}
            
            
            df_topwords = pd.DataFrame( topword_list, columns = ["Word", "Frequency"])
            df_topwords  = df_topwords.style.hide_index()
            
            
            st.subheader('Word Frequency')
            st.write(df_topwords)
            
            my_placeholder = st.empty()
            my_placeholder.text('')
            
            wc = WordCloud(background_color = 'white').fit_words(topword_dict)
            st.image(wc.to_array())
            
            # wc_stopword = st.multiselect('Multiselect', [x[0] for x in topword_list])
    
    if nav == "Keyword Finder":
        st.title('Keyword Finder')
        
        col1, col2 = st.columns([1,2])
        
        with col1:
            with st.form(key='my_form'):
                data_sheet = st.file_uploader('File uploader', type = ['xlsx'])
                excel_sheet_name = st.text_input("Please enter sheet name here")
                col_name = st.text_input('Please enter column name')
                topic_sheet = st.file_uploader('Topic uploader', 
                                               type = ['xlsx'],
                                               help = "Please upload an excel file with header columns and on Sheet1")
                submit_button = st.form_submit_button(label='Process Data')
                
            if data_sheet is not None:
                if data_sheet.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    df_raw = pd.read_excel(data_sheet, sheet_name = excel_sheet_name)
                    df_topics = pd.read_excel(topic_sheet, sheet_name = "Sheet1")
                    
                    df_clean = clean_data(df_raw, col_name)
                    list_df = convertdf_to_dataword(df_clean, col_name)
             
                    output_dict = topic_search(df_topics, list_df)
                    trans_to_df_dict = { x:pd.Series(output_dict[x]) for x in output_dict}
                    output_df = pd.DataFrame.from_dict(trans_to_df_dict)
                    plot_df = output_df.count().rename_axis('Topics').reset_index(name='Count')
                    plot_df = plot_df[plot_df["Count"]>0]
                    @st.cache
                    def convert_df(df):
                        # IMPORTANT: Cache the conversion to prevent computation on every rerun
                        return df.to_csv(index=False).encode('utf-8')
    
                    csv = convert_df(output_df)
                    st.download_button(
                            label="Download data as CSV",
                            data=csv,
                            file_name='Topics from Keywords.csv',
                            mime='text/csv')
                    
                    with col2:
                        bar_chart = px.bar(
                            data_frame = plot_df,
                            x = 'Count',
                            y = 'Topics',
                            orientation = 'h',
                            title = 'Topic Frequency')
                        bar_chart.update_traces(texttemplate='%{x:0s}', textposition='outside')
                        bar_chart.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                        bar_chart.update_layout( yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(bar_chart)
        
    if nav == "Topic Modelling":
        st.title('Topic Modelling')
        
        col1, col2 = st.columns([1,2])
        
        
        with col1:
            with st.form(key='my_form'):
                data_sheet = st.file_uploader('File uploader', 
                                              type = ['xlsx'], 
                                              help = "Please upload an excel file with header columns")
                excel_sheet_name = st.text_input("Please enter sheet name here")
                col_name = st.text_input('Please enter column name')
                
                pos_tags = st.multiselect(
                        'Select included Parts of Speech tags',
                        ['Proper Noun', 'Noun', 'Verb', 'Adjective', "Adverb"],
                        ['Proper Noun', 'Noun', 'Verb', 'Adjective', "Adverb"])
                included_pos = [pos_tag_dict[x] for x in pos_tags]
                bigram_check = st.radio("Choose an n-gram model",("Bigram","Trigram"))
                
                topic_count = st.slider('Number of Topics', min_value=2, max_value=15)
                
                submit_data = st.form_submit_button(label='Process Data')
            
        if data_sheet is not None:
            if data_sheet.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df_raw = pd.read_excel(data_sheet, sheet_name = excel_sheet_name)
            
            clean_df = clean_data(df_raw, col_name)
            lem_df = clean_df.apply(lemmatization, args= [included_pos])
    
            data_words = convertdf_to_dataword(lem_df, col_name)
            
            bigrams_phrases = gensim.models.Phrases(data_words, min_count=5, threshold = 50)
            trigrams_phrases = gensim.models.Phrases(bigrams_phrases[data_words], threshold=50)

            bigram = gensim.models.phrases.Phraser(bigrams_phrases)
            trigram = gensim.models.phrases.Phraser(trigrams_phrases)

            def make_bigram(texts): 
                return [bigram[doc] for doc in texts]

            def make_trigram(texts):
                return [trigram[bigram[doc]] for doc in texts]
            
            data_bigrams = make_bigram(data_words)
            data_bigrams_trigrams = make_trigram(data_bigrams)
            
            if bigram_check == "Bigram":
                dict_gram = collect_bigrams(data_bigrams)
                gram_df = list_to_df(data_bigrams, col_name)
                
            elif bigram_check == "Trigram":
                dict_gram = collect_bigrams(data_bigrams_trigrams)
                gram_df = list_to_df(data_bigrams_trigrams, col_name)
            
            else:
                gram_df = lem_df
                
            dict_gram_df = pd.DataFrame.from_dict(dict_gram)
            dict_gram_tdf = dict_gram_df.T.reset_index()
            dict_gram_tdf.rename(columns = {dict_gram_tdf.columns[0]:bigram_check, dict_gram_tdf.columns[1]:'Count'},inplace =True)
            
            bar_chart = px.bar(
                    data_frame = dict_gram_tdf,
                    x = 'Count',
                    y = bigram_check,
                    orientation = 'h')
            
            with col2:
                st.plotly_chart(bar_chart)
            
            
            cv = CountVectorizer(stop_words='english')
            data_cv = cv.fit_transform(gram_df[col_name])
            data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
            data_dtm.index = lem_df[col_name]
            
            tdm = data_dtm.transpose()
            sparse_counts = scipy.sparse.csr_matrix(tdm)
            corpus = matutils.Sparse2Corpus(sparse_counts)
            
            id2word = dict((v, k) for k, v in cv.vocabulary_.items())
            word2id = dict((k, v) for k, v in cv.vocabulary_.items())
            d = corpora.Dictionary()
            d.id2token = id2word
            d.token2id = word2id
            
            lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                               id2word = d,
                                               num_topics = topic_count,
                                               random_state = 100,
                                               update_every =1,
                                               chunksize = 100,
                                               passes = 10,
                                               alpha = "auto")

            vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, d, mds = "mmds", R=30)
            
            html_string = pyLDAvis.prepared_data_to_html(vis)
            from streamlit import components
            components.v1.html(html_string, width = 1300, height = 800)
    
if __name__ == '__main__':
    main()
# In[ ]:



