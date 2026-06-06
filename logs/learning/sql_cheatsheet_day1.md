# Day 1 SQL 速查表

## 查询数据

```sql
SELECT *
FROM table_name;
```

`*` 表示全部列。实际商业项目中，应逐步改为明确列名。

## 只看前几行

```sql
SELECT *
FROM table_name
LIMIT 10;
```

## 筛选记录

```sql
SELECT *
FROM table_name
WHERE quantity >= 10;
```

常用比较符号：

| 符号 | 含义 |
|---|---|
| `=` | 等于 |
| `<>` 或 `!=` | 不等于 |
| `>`、`>=` | 大于、大于等于 |
| `<`、`<=` | 小于、小于等于 |
| `BETWEEN a AND b` | 位于范围内 |
| `IN (...)` | 位于指定集合内 |
| `IS NULL` | 缺失值 |

## 排序

```sql
SELECT *
FROM table_name
ORDER BY unit_price DESC;
```

- `ASC`：从小到大，默认。
- `DESC`：从大到小。

## 去重列表

```sql
SELECT DISTINCT country
FROM table_name;
```

## 计算

```sql
SELECT quantity * unit_price AS line_value
FROM table_name;
```

`AS` 用于给结果列一个容易理解的名字。

## 聚合函数

```sql
SELECT
    COUNT(*) AS row_count,
    SUM(quantity) AS total_quantity,
    AVG(unit_price) AS average_price,
    MIN(unit_price) AS minimum_price,
    MAX(unit_price) AS maximum_price
FROM table_name;
```

## 分组

```sql
SELECT
    country,
    COUNT(*) AS row_count
FROM table_name
GROUP BY country;
```

选择结果中出现了普通字段和聚合函数时，普通字段通常需要放在 `GROUP BY` 中。

## WHERE 与 HAVING

```sql
-- WHERE：在分组之前筛选行
SELECT country, COUNT(*) AS row_count
FROM table_name
WHERE unit_price > 0
GROUP BY country;

-- HAVING：在分组之后筛选组
SELECT country, COUNT(*) AS row_count
FROM table_name
GROUP BY country
HAVING COUNT(*) > 3;
```

## NULL

```sql
SELECT *
FROM table_name
WHERE customer_id IS NULL;
```

不要写：

```sql
WHERE customer_id = NULL;
```

## COUNT 的区别

```sql
COUNT(*)                    -- 所有行
COUNT(customer_id)          -- customer_id 非空的行
COUNT(DISTINCT customer_id) -- 不重复的已知客户
```
