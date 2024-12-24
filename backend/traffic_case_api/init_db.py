import asyncio
import os
from pathlib import Path
from app.data_manager import CaseDataManager
from app.case_generator import TrafficCaseGenerator

async def init_database():
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Ensure proper permissions for data directory
    os.system("chmod 777 data")
    
    # Generate 500 cases using the case generator
    generator = TrafficCaseGenerator()
    cases = generator.generate_cases(1, 500)
    
    # Initialize data manager and save cases
    data_manager = CaseDataManager()
    await data_manager.save_cases(cases)
    print(f"Database initialized with {len(cases)} cases.")

if __name__ == "__main__":
    asyncio.run(init_database())
