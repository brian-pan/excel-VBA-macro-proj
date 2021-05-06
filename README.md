## Solution Set for Shopify Intern
#### Name: Zifan Pan
#### Date: May 6, 2021

## Question 1


```python
import numpy as np
import pandas as pd
```


```python
df = pd.read_csv('question_1_data.csv')
```


```python
#check the format of data
df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>shop_id</th>
      <th>user_id</th>
      <th>order_amount</th>
      <th>total_items</th>
      <th>payment_method</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>53</td>
      <td>746</td>
      <td>224</td>
      <td>2</td>
      <td>cash</td>
      <td>2017-03-13 12:36:56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>92</td>
      <td>925</td>
      <td>90</td>
      <td>1</td>
      <td>cash</td>
      <td>2017-03-03 17:38:52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>44</td>
      <td>861</td>
      <td>144</td>
      <td>1</td>
      <td>cash</td>
      <td>2017-03-14 4:23:56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>18</td>
      <td>935</td>
      <td>156</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-26 12:43:37</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>18</td>
      <td>883</td>
      <td>156</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-01 4:35:11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>58</td>
      <td>882</td>
      <td>138</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-14 15:25:01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>87</td>
      <td>915</td>
      <td>149</td>
      <td>1</td>
      <td>cash</td>
      <td>2017-03-01 21:37:57</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>22</td>
      <td>761</td>
      <td>292</td>
      <td>2</td>
      <td>cash</td>
      <td>2017-03-08 2:05:38</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>64</td>
      <td>914</td>
      <td>266</td>
      <td>2</td>
      <td>debit</td>
      <td>2017-03-17 20:56:50</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>52</td>
      <td>788</td>
      <td>146</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-30 21:08:26</td>
    </tr>
  </tbody>
</table>
</div>



### part a.

```python
#find the original average order value
df['order_amount'].mean()
```




    3145.128



The average order value is 3145.128, which is relatively higher than expected. From the first ten data above, we notice that the order amount for each store is significantly less than the average value of 3145.128. Thus there must be some abnormal outliers which are higher than 3145. 


```python
#Filter out data that are above 3145
df[df['order_amount'] > 3145]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>shop_id</th>
      <th>user_id</th>
      <th>order_amount</th>
      <th>total_items</th>
      <th>payment_method</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-07 4:00:00</td>
    </tr>
    <tr>
      <th>60</th>
      <td>61</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-04 4:00:00</td>
    </tr>
    <tr>
      <th>160</th>
      <td>161</td>
      <td>78</td>
      <td>990</td>
      <td>25725</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-12 5:56:57</td>
    </tr>
    <tr>
      <th>490</th>
      <td>491</td>
      <td>78</td>
      <td>936</td>
      <td>51450</td>
      <td>2</td>
      <td>debit</td>
      <td>2017-03-26 17:08:19</td>
    </tr>
    <tr>
      <th>493</th>
      <td>494</td>
      <td>78</td>
      <td>983</td>
      <td>51450</td>
      <td>2</td>
      <td>cash</td>
      <td>2017-03-16 21:39:35</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4646</th>
      <td>4647</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-02 4:00:00</td>
    </tr>
    <tr>
      <th>4715</th>
      <td>4716</td>
      <td>78</td>
      <td>818</td>
      <td>77175</td>
      <td>3</td>
      <td>debit</td>
      <td>2017-03-05 5:10:44</td>
    </tr>
    <tr>
      <th>4868</th>
      <td>4869</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-22 4:00:00</td>
    </tr>
    <tr>
      <th>4882</th>
      <td>4883</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-25 4:00:00</td>
    </tr>
    <tr>
      <th>4918</th>
      <td>4919</td>
      <td>78</td>
      <td>823</td>
      <td>25725</td>
      <td>1</td>
      <td>cash</td>
      <td>2017-03-15 13:26:46</td>
    </tr>
  </tbody>
</table>
<p>63 rows × 7 columns</p>
</div>



We notice that shop #42 selled 2000 items for a total amount of 704000, which is not normal compared to other stores. Besides, shop #78 selled one pair of sneakers for 25725, which is not a common situation too. The data from these stores should not be included if we are calculating the average order value under common situations.  


```python
#find which stores provided the abnormal data point,
# and count how many times the abnormal data appeared
df[df['order_amount'] > 3145].groupby('shop_id').count()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>order_amount</th>
      <th>total_items</th>
      <th>payment_method</th>
      <th>created_at</th>
    </tr>
    <tr>
      <th>shop_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
    </tr>
    <tr>
      <th>78</th>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
    </tr>
  </tbody>
</table>
</div>



Based on the above table obtained, we notice that there are only two shops which gives the outliers: shop #42 and shop #78, and they showed 17 and 46 times respectively. 


```python
#remove these data points
df2 = df[df['order_amount'] < 3145]
```


```python
df2
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>shop_id</th>
      <th>user_id</th>
      <th>order_amount</th>
      <th>total_items</th>
      <th>payment_method</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>53</td>
      <td>746</td>
      <td>224</td>
      <td>2</td>
      <td>cash</td>
      <td>2017-03-13 12:36:56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>92</td>
      <td>925</td>
      <td>90</td>
      <td>1</td>
      <td>cash</td>
      <td>2017-03-03 17:38:52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>44</td>
      <td>861</td>
      <td>144</td>
      <td>1</td>
      <td>cash</td>
      <td>2017-03-14 4:23:56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>18</td>
      <td>935</td>
      <td>156</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-26 12:43:37</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>18</td>
      <td>883</td>
      <td>156</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-01 4:35:11</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>4996</td>
      <td>73</td>
      <td>993</td>
      <td>330</td>
      <td>2</td>
      <td>debit</td>
      <td>2017-03-30 13:47:17</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>4997</td>
      <td>48</td>
      <td>789</td>
      <td>234</td>
      <td>2</td>
      <td>cash</td>
      <td>2017-03-16 20:36:16</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>4998</td>
      <td>56</td>
      <td>867</td>
      <td>351</td>
      <td>3</td>
      <td>cash</td>
      <td>2017-03-19 5:42:42</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>4999</td>
      <td>60</td>
      <td>825</td>
      <td>354</td>
      <td>2</td>
      <td>credit_card</td>
      <td>2017-03-16 14:51:18</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>5000</td>
      <td>44</td>
      <td>734</td>
      <td>288</td>
      <td>2</td>
      <td>debit</td>
      <td>2017-03-18 15:48:18</td>
    </tr>
  </tbody>
</table>
<p>4937 rows × 7 columns</p>
</div>




```python
#check if there are any missing points
df2.isnull().sum()
```




    order_id          0
    shop_id           0
    user_id           0
    order_amount      0
    total_items       0
    payment_method    0
    created_at        0
    dtype: int64



### part b.
In reality, the price for a pair of normal sneakers would be around 90-150 dollars depends on the specific model. Therefore, it would be reasonable if the metric is dollar (USD/CAD/etc.). 

Now find the average order value (AOV):


```python
#find AOV
df2['order_amount'].mean() 
```




    302.58051448247926




```python
df2['order_amount'].count() #total number of transactions
```




    4937




```python
df2['total_items'].sum() #average number of sneakers sold per store
```




    9848




```python
df2['shop_id'].nunique() #total number of stores left
```




    99




```python
df2['total_items'].mean() #average number of sneakers sold per transaction
```




    1.9947336439133077




```python
df2['order_amount'].describe() #statistics of AOV
```




    count    4937.000000
    mean      302.580514
    std       160.804912
    min        90.000000
    25%       163.000000
    50%       284.000000
    75%       387.000000
    max      1760.000000
    Name: order_amount, dtype: float64




```python
df2['user_id'].nunique() #how many individuals has made at least one purchase
```




    300




```python
df2['user_id'].value_counts().max() #maximum number of transaction times per user 
```




    28




```python
df2['user_id'].value_counts().min() #minimum number of transaction times per user 
```




    7




```python
df[df['order_amount'] > 3145].drop(['order_id','user_id'], axis=1).groupby('shop_id').sum() #large transaction amounts
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_amount</th>
      <th>total_items</th>
    </tr>
    <tr>
      <th>shop_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>11968000</td>
      <td>34000</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2263800</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>


### part c.
In short, the average order value across all 4937 transactions was 302.58 dollars, meaning on average Shopify can expected the shoe stores to sell 2 sneakers and have 302.58 dollars of income per transaction. During this 30-day window, all 99 stores sold a total number of 9848 sneakers, where the lowest transaction amount was 90 dollars, and the highest transaction amount was 1760 dollars. The median of all transactions was 284 dollars, but the mean was 302.58 dollars, so that the data is right-skewed, which means the stores should expect most of consumers spent less than 302.58 dollars each time they purchase snearkers. Besides, not all consumers has made a purchase, and some consumers purchased multiple times. Some user(s) made at most 28 times of transactions and some user(s) made at least 7 times of transactions. Lastly, regrading the large transactions, under the premise that the data is true, it would be benefical if store #42 and store #78 can seek the long-term cooperations with them. 


## Question 2
### a.
```SQL
SELECT COUNT(*)
FROM Orders
WHERE ShipperID = '1';
```
Solution: The total number of orders shipped by Speedy Express is 54.
### b.
```SQL
SELECT E.EmployeeID, LastName, COUNT(O.OrderID) AS TotalNum
FROM Employees AS E, Orders AS O
WHERE E.EmployeeID = O.EmployeeID
GROUP BY E.EmployeeID
ORDER BY TotalNum DESC;
```
Solution: The last name of the employee with the most orders is Peacock.
### c.
```SQL
SELECT P.ProductID, P.ProductName, SUM(OD.Quantity) AS TotalQuantity
FROM Products AS P, Orders AS O, OrderDetails AS OD, Customers AS C
WHERE P.ProductID = OD.ProductID
AND O.CustomerID = C.CustomerID
AND O.OrderID = OD.OrderID
AND Country = 'Germany'
GROUP BY 1,2
ORDER BY TotalQuantity DESC;
```
Solution: The Product ordered the most by customers in Germany is the Boston Crab Meat with Product ID of 40. 