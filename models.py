from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Fortune500(Base):
    __tablename__ = 'fortune_500'

    id = Column(Integer, primary_key=True)
    Name = Column(String(255))
    Rank = Column(Integer)
    Revenues = Column(String(10), name='Revenues')
    Revenue_Percent_Change = Column(String(10), name = 'Revenue Percent Change')
    Profits = Column(String(10), name = 'Profits ($M)')
    Profits_Percent_Change = Column(String(10), name = 'Profits Percent Change')
    Assets = Column(String(100))
    Newcomer_to_the_Global_500 = Column(String(10), name = 'Newcomer to the Global 500')
    Employees = Column(String(10))
    Dropped_in_Rank = Column(String(10), name = 'Dropped in Rank')
    Gained_in_Rank = Column(String(10), name = 'Gained in Rank')
    Sector = Column(String(255))
    Industry = Column(String(255))
    Country_Territory = Column(String(255), name = 'Country / Territory')
    Headquarters_City = Column(String(255), name = 'Headquarters City')
    Headquarters_State = Column(String(255), name = 'Headquarters State')
    Years_on_Global_500_List = Column(String(255), name = 'Years on Global 500 List')
    Profitable = Column(String(10))
    Worlds_Most_Admired_Companies = Column(String(10), name = "World's Most Admired Companies")
    Female_CEO = Column(String(10), name = 'Female CEO')
    Growth_in_Jobs = Column(String(10), name = 'Growth in Jobs')
    Change_the_World = Column(String(10), name = 'Change the World')
    Fastest_Growing_Companies = Column(String(10), name = 'Fastest Growing Companies')
    Fortune_500 = Column(String(10), name = 'Fortune 500')
    Best_Companies = Column(String(10), name = 'Best Companies')
    Non_US_Companies = Column(String(10), name = 'Non-U.S. Companies')
    Change_in_Rank = Column(String(10), name = 'Change in Rank')

    
class ExecutionStatus(Base):
    __tablename__ = 'execution_status'

    id = Column(Integer, primary_key=True)
    script_name = Column(String(255), unique=True)
    executed = Column(Boolean, default=False)

    def __repr__(self):
        return f"<ExecutionStatus(script_name='{self.script_name}', executed={self.executed})>"