Declare @t varchar (1024)
Declare tbl_cur cursor for  
select TABLE_NAME from INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'

OPEN tbl_cur

FETCH NEXT  from tbl_cur INTO @t

WHILE @@FETCH_STATUS = 0
BEGIN
EXEC ('TRUNCATE TABLE '+ @t)
FETCH NEXT  from tbl_cur INTO @t
END

CLOSE tbl_cur
DEALLOCATE tbl_Cur

