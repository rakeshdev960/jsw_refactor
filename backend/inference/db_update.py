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
        if query.strip().lower().startswith(('insert', 'update', 'delete', 'alter', 'create', 'drop')):
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

def check_if_table_exists(table_name):
    """Check if the table exists in the database"""
    query = f"""SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
             WHERE TABLE_SCHEMA = 'jsw_inv_mng' 
             AND TABLE_NAME = '{table_name}'"""
    
    result = execute_sql(query)
    return result and len(result) > 0

def create_bag_movement_table():
    """Create the bag_movement table if it doesn't exist"""
    if check_if_table_exists('bag_movement'):
        print("Table bag_movement already exists!")
        return True
    
    print("Creating bag_movement table...")
    create_query = """
    CREATE TABLE bag_movement (
        id INT AUTO_INCREMENT PRIMARY KEY,
        cluster_id INT NOT NULL,
        timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        movement_type VARCHAR(10) NOT NULL,
        quantity INT NOT NULL DEFAULT 0,
        FOREIGN KEY (cluster_id) REFERENCES cluster(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    success = execute_sql(create_query)
    
    if success:
        print("Successfully created bag_movement table.")
        return True
    else:
        print("Failed to create bag_movement table.")
        return False

if __name__ == "__main__":
    print("Updating database with bag movement tracking...")
    success = create_bag_movement_table()
    
    if success:
        print("Database schema update completed successfully.")
    else:
        print("Database schema update failed.")
        sys.exit(1)
