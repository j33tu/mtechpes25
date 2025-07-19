import mysql.connector
from mysql.connector import Error

def create_database():
    try:
         
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password=""
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
           
            cursor.execute("CREATE DATABASE IF NOT EXISTS dirtyvehicleplate_2025")
            print("Database 'dirtyvehicleplate_2025' created or already exists.")
            
            
            cursor.execute("USE dirtyvehicleplate_2025")
            
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) NOT NULL UNIQUE,
                    email VARCHAR(100) NOT NULL UNIQUE,
                    password VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("Table 'users' created or already exists.")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    video_path VARCHAR(255) NOT NULL,
                    
                    plate_image_path VARCHAR(255) NOT NULL,
                    plate_number VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    run_id VARCHAR(255) NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            print("Table 'predictions' created or already exists.")
            
            connection.commit()
            
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

if __name__ == "__main__":
    create_database()