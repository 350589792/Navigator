from data_loader import DataLoader

def main():
    try:
        loader = DataLoader("data")
        loader.validate_files()
        print("All required files validated successfully!")
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
