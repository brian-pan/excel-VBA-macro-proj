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
Solution: The Product ordered the most by customers in Germany is the Boston Crab Meat with a total amount of 160.
