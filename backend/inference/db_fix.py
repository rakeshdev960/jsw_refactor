import mysql.connector
import sys

# Database connection parameters
DB_CONFIG = {
    'user': 'root',
    'password': 'admin',
    'host': 'localhost',
    'database': 'jsw_inv_mng'
}

def execute_sql(query):
    """Execute a SQL query and return the results"""
    conn = None
    cursor = None
    try:
        # Establish connection
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Execute query
        print(f"Executing SQL: {query}")
        cursor.execute(query)
        
        # Commit if needed
        if query.strip().lower().startswith(('insert', 'update', 'delete', 'alter', 'create')):
            conn.commit()
            print("Changes committed.")
        
        # Return results if it's a SELECT
        if query.strip().lower().startswith('select'):
            return cursor.fetchall()
        
        return True
    
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        return False
    
    finally:
        # Close cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def check_if_column_exists():
    """Check if the capacity column already exists in the cluster table"""
    query = """SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
             WHERE TABLE_SCHEMA = 'jsw_inv_mng' 
             AND TABLE_NAME = 'cluster' 
             AND COLUMN_NAME = 'capacity'"""
    
    result = execute_sql(query)
    return result and len(result) > 0

def add_capacity_column():
    """Add the capacity column to the cluster table"""
    print("Checking if capacity column exists...")
    
    if check_if_column_exists():
        print("Capacity column already exists!")
        return True
    
    print("Adding capacity column to the cluster table...")
    
    # Alter table to add capacity column with default value
    alter_query = "ALTER TABLE cluster ADD COLUMN capacity INT DEFAULT 1000;"
    success = execute_sql(alter_query)
    
    if success:
        print("Successfully added capacity column with default value 1000.")
        return True
    else:
        print("Failed to add capacity column.")
        return False

if __name__ == "__main__":
    success = add_capacity_column()
    if success:
        print("Database schema successfully updated.")
    else:
        print("Failed to update database schema.")
        sys.exit(1)
