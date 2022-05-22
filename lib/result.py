class Result:
    def __init__(self) -> None: 
        self.velocity = 0
        self.residuals = 0
        self.positions = []
        self.time_instants = []
        self.ui_flow_rate_result_value = 0.0
        self.start = 0
        self.head = ''
        self.slope = 0 
        self.intercept = 0
        self.predicted_positions = []
        self.ui_flow_rate_result_dynm = 0
        self.mean_position = None
        self.positions_nomber = 0
        self.time_instants_number = 0
        self.time_instants_dynm = 0
        self.flow_rate_dynm = 0