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
    
class SalesData(Base):
    __tablename__ = 'sales_data'
    
    id = Column(Integer, primary_key=True)
    opportunity_id = Column(String(255), unique=True)
    technology_primary = Column(String(150))
    city = Column(String(50))
    b2b_sales_medium = Column(String(150))
    sales_velocity = Column(Integer)
    opportunity_status = Column(String(10))
    sales_stage_iterations = Column(Integer)
    opportunity_size_usd = Column(Integer)
    client_revenue_sizing = Column(String(150))
    client_employee_sizing = Column(String(150))
    business_from_client_last_year = Column(String(150))
    compete_intel = Column(String(150))
    opportunity_sizing = Column(String(150))
    

class B2BSalesData(Base):
    __tablename__ = 'b2b_sales_data'
    
    id = Column(Integer, primary_key=True)
    Product = Column(String(100))
    Seller = Column(String(100))
    Authority = Column(String(50))
    Comp_size = Column(String(150))
    Competitors = Column(String(10))
    Purch_dept = Column(String(10))
    Partnership = Column(String(10))
    Budgt_alloc = Column(String(10))
    Forml_tend = Column(String(10))
    RFI = Column(String(10))
    RFP = Column(String(10))
    Growth = Column(String(50))
    Posit_statm = Column(String(50))
    Source = Column(String(50))
    Client = Column(String(50))
    Scope = Column(String(50))
    Strat_deal = Column(String(50))
    Cross_sale = Column(String(10))
    Up_sale = Column(String(10))
    Deal_type = Column(String(50))
    Needs_def = Column(String(50))
    Att_t_client = Column(String(50))
    Status = Column(String(10))
    
class ExecutionStatus(Base):
    __tablename__ = 'execution_status'

    id = Column(Integer, primary_key=True)
    script_name = Column(String(255), unique=True)
    executed = Column(Boolean, default=False)

    def __repr__(self):
        return f"<ExecutionStatus(script_name='{self.script_name}', executed={self.executed})>"