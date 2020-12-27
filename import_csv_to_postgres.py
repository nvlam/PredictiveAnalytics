# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import psycopg2
import sqlalchemy
import pandas as pd


# %%
from sqlalchemy import create_engine
cnx = create_engine('postgresql://postgres:2413@localhost:5432/test_db')
conn=cnx.connect()


# %%
df15=pd.read_csv('POS_data_2015.csv') #change file path to relevant one on your computer


# %%
df15.head() # check for data read


# %%
df15['year']=2015 # add year column and fill with relevant year


# %%
# insert data to postgres table named 'post_data',
df15.to_sql('post_data',conn,schema='common',if_exists='append',index_label='index')


# %%
df16=pd.read_csv('POS_data_2016.csv')
#df16.drop(columns=['Unnamed: 0'])
df16['year']='2016'


# %%
df16.to_sql('post_data',conn,schema='common',if_exists='append',index_label='index')


# %%
df17=pd.read_csv('POS_data_2017.csv')
#df17.drop(columns=['Unnamed: 0'])
df17['year']='2017'
df17.to_sql('post_data',conn,schema='common',if_exists='append',index_label='index')


# %%
df18=pd.read_csv('POS_data_2018.csv')


# %%
df18.head()


# %%
df18['year']='2018'


# %%
#be careful, 'invoice_close' column names are not consistent in 4 POS_data_file.
#have to rename column name to 'invoice_close' to be the same as they are in 'POS_data_2015.csv,POS_data_2016.csv,POS_data_2017.csv'
#df18.rename(columns={'invoice_closed':'invoice_close'})


# %%
df18.head()


# %%
#be careful, column name of the csv file need to be changed to 'invoice_close' to be consistent.
df18.to_sql('post_data',conn,schema='common',if_exists='append',index_label='index')


# %%
df = pd.read_sql_query('''SELECT * FROM common.post_data where year=2018 LIMIT 10;''',conn)
df # check data if data inserted.


# %%
sql_str='''select p.article_number,p.kind,p.invoice_close,p.invoice_opened,
p.guests,p.group,p.invoice,p.ticket,p.table,p.time,
a.price, a.year
from common.articles a inner join common.post_data p 
on (a.article_number=p.article_number and
 a.year=p.year);'''

df1 = pd.read_sql_query(sql_str,conn)


# %%
df1.head()


# %%
conn.close()


# %%



