import numpy as np

CONTEXT_LOW = -5.0
CONTEXT_HIGH = 5.0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Auction:
    def __init__(self, rng, agents, CTR_param, item_features, context_dim, context_dist, horizon, budget):
        # auction = Auction(rng, agents, CTR_param, run2item_features[run], context_dim, context_dist, horizon, budget)
        super().__init__()
        self.rng = rng
        self.agents = agents
        self.num_agents = len(agents)

        self.CTR_param = CTR_param
        self.item_features = item_features      # num_agent x feature_dim
        self.context_dim = context_dim
        
        self.context_dist = context_dist # Gaussian, Bernoulli, Uniform
        self.gaussian_var = 1.0
        self.bernoulli_p = 0.5

        self.horizon = horizon
        self.budget = budget
    
    def generate_context(self):
        if self.context_dist=='Gaussian':
            context = self.rng.normal(0.0, 1.0, size=self.context_dim)
        elif self.context_dist=='Bernoulli':
            # N번의 시행 중 발생확률 p일 때 발생횟수 나타내는 확률변수 X~B(n,p)
            # 여기선, N=1이므로 X=0 or 1이다. 즉, context는 one-hot vector
            context = self.rng.binomial(1, self.bernoulli_p, size=self.context_dim)
        else:
            # low = -1, high = 1인 uniform distribution
            context = self.rng.uniform(-1.0, 1.0, size=self.context_dim)
        return np.clip(context, CONTEXT_LOW, CONTEXT_HIGH)
        # np.clip(array, min, max) : array 내의 모든 element들 min보다 작으면 min으로, max보다 크면 max로 바꾸는 함수
    
    def reset(self):
        self.context = self.generate_context()
        self.remaining_steps = self.horizon * np.ones((self.num_agents))
        self.remaining_budget = self.budget * np.ones((self.num_agents))

        return np.concatenate([np.tile(self.context,(self.num_agents,1)), self.remaining_budget.reshape(-1,1), self.remaining_steps.reshape(-1,1)],axis=1), {}
        # 여기서 tile 결과는 self.context vector를 행으로 num_agents만큼 아래로 복붙 
        # np.tile(a, (2, 2, 3))
        '''a = np.array([5, 6])
           b = np.tile(a, (2,2,3))
           b
           array([[[5, 6, 5, 6, 5, 6],
                   [5, 6, 5, 6, 5, 6]],
                   
                  [[5, 6, 5, 6, 5, 6],
                   [5, 6, 5, 6, 5, 6]]]), shape = (2, 2, 6)'''
                   
        # concat 결과는 행방향 tile된 각 agent별 context에 열방향으로 각 agent별 remaining step과 budget 연결한 것
        # reshape(-1,1)은 2차원 배열로 변경하기 위해 필요함

    def step(self, actions):
        CTR = []
        for j in range(self.num_agents):
            CTR.append(sigmoid(self.context @ self.CTR_param @ self.item_features[j] / np.sqrt(self.context_dim)).item())
            # self.item_features는 run2item : shape(num_runs, num_agents, feature_dim)이므로
            # self.item_feature[j]는 각 agent의 feature_encoding vector(a_i)이며 행렬곱 결과는 scalar
            # numpy.ndarray.item(*args) : *args = None인 경우 np객체를 파이썬 표준 스칼라 객체로 복사하여 반환
            # args가 int거나 int_types tuple인 경우 ndarray에 대한 flat index로 해석된다.
        outcome = self.rng.binomial(1, p=CTR)

        win = np.zeros((self.num_agents))
        win[np.argmax(actions)] = 1
        reward =  outcome * win

        info = {
            'win' : win,
            'outcome' : outcome
        }
        self.remaining_budget -= win * np.array(actions)
        self.remaining_steps -= 1
        self.context = self.generate_context()
        return np.concatenate([np.tile(self.context,(self.num_agents,1)), self.remaining_budget.reshape(-1,1), self.remaining_steps.reshape(-1,1)],axis=1), \
                                reward, np.logical_or(self.remaining_budget<1e-3, self.remaining_steps==0), info