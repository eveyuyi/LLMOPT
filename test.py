

from pyomo.environ import *

class PropertyInvestmentModel:
    def __init__(self):
        self.model = ConcreteModel()
        
        # Sets
        self.model.I = Set(initialize=[1, 2, 3, 4])
        
        # Parameters
        self.model.Income = Param(self.model.I, initialize={1: 12500, 2: 35000, 3: 23000, 4: 100000})
        self.model.Cost = Param(self.model.I, initialize={1: 1.5e6, 2: 2.1e6, 3: 2.3e6, 4: 4.2e6})
        self.model.B = Param(initialize=7e6)
        
        # Variables
        self.model.x = Var(self.model.I, domain=Binary)
        
        # Objective
        def obj_expression(model):
            return sum(model.Income[i] * model.x[i] for i in model.I)
        self.model.Objective = Objective(rule=obj_expression, sense=maximize)
        
        # Constraints
        def budget_constraint(model):
            return sum(model.Cost[i] * model.x[i] for i in model.I) <= model.B
        self.model.BudgetConstraint = Constraint(rule=budget_constraint)
        
        def exclusive_purchase_constraint(model):
            return model.x[4] + model.x[3] <= 1
        self.model.ExclusivePurchaseConstraint = Constraint(rule=exclusive_purchase_constraint)

    def solve(self):
        solver = SolverFactory('glpk')
        results = solver.solve(self.model, tee=False)
        self.display_results()

    def display_results(self):
        print("Optimal Solution:")
        for i in self.model.I:
            if self.model.x[i]() != 0:
                print(f"Property {i} purchased.")
        print(f"Total Annual Income: ${self.model.Objective()}")

def main():
    investment_model = PropertyInvestmentModel()
    investment_model.solve()

if __name__ == "__main__":
    main()
