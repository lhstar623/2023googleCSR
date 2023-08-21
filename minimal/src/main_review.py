import argparse 
import json
import numpy as np
import os
import shutil
from copy import deepcopy
from tqdm import tqdm
import time

import numpy as np
from gym.spaces import Dict, Box

from Agent import *
from Allocator import *
from Auction import Auction
from Bidder import * 
from plot import *

class Buffer: #경험 리플레이 버퍼 구현 (에이전트가 경험 저장, 무작위 샘플링으로 학습에 사용)
    def __init__(self): #아래 attributes를 빈 리스트로 초기화
        self.states = [] #에이전트의 현재 상태 저장
        self.items = [] #bidding에서 선택한 항목에 대한 정보 저장
        self.biddings = [] #bidding에서 입찰 정보 저장
        self.rewards = [] #에이전트가 받은 보상 저장 
        self.next_states = [] #에이전트의 다음 상태 저장
        self.d = [] #에피소드 종료 여부 저장
        self.wins = [] #에이전트가 경매에서 낙찰에 성공했는지 여부 저장
        self.outcomes = [] #에이전트의 경매 결과 저장

    def append(self, state, item, bidding, reward, next_state, done, win, outcome):
        #경험을 버퍼에 추가하는 method, 한 타임 스텝에서 받은 정보를 인자로 받아 버퍼에 저장
        self.states.append(state)
        self.items.append(item)
        self.biddings.append(bidding)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.d.append(done)
        self.wins.append(win)
        self.outcomes.append(outcome)
    
    def numpy(self):
        #버퍼의 모든 데이터를 numpy 배열 형태로 반환
        return np.array(self.states), np.array(self.items), np.array(self.biddings), np.array(self.rewards),\
              np.array(self.next_states), np.array(self.d, dtype=bool), np.array(self.wins, dtype=bool), np.array(self.outcomes, dtype=bool)
    
    def sample(self, batch_size): 
        #버퍼에서 무작위로 지정한 배치 크기만큼 데이터를 샘플링하여 반환
        if batch_size > len(self.states): # 모든 데이터 활용
            self.numpy()
        else: 
            #np.random.choice 함수로 batch_size만큼 인덱스를 각각 샘플링함
            #replace=false: 중복 허용하지 않고 샘플링->독립성 보장(기본값은 중복 허용, true)
            inds = np.random.choice(len(self.states), batch_size, replace = False)
            return np.array([self.states[idx] for idx in inds]), np.array([self.items[idx] for idx in inds]), np.array([self.biddings[idx] for idx in inds]), \
                np.array([self.rewards[idx] for idx in inds]), np.array([self.next_states[idx] for idx in inds]), np.array([self.d[idx] for idx in inds], dtype=bool)

def draw_features(rng, num_runs, feature_dim):
    #랜덤넘버생성기(rng)를 사용하여 각 run에 대해 아이템의 feature과 value를 생성하는 함수->랜덤성 보장
    #rng = 랜덤넘버 생성기, num_runs: 반복할 횟수, feature_dim: 아이템의 특징 차원    
    run2item_features = {}
    run2item_values = {}

    for run in range(num_runs):
        temp = []
        for k in range(training_config['num_items']): #경매에 사용되는 아이템의 개수만큼 반복
            feature = rng.normal(0.0, 1.0, size=feature_dim) #평균이 0이고 표준편차가1인 정규분포로부터 랜덤 생성
            temp.append(feature) #생성된 특징을 임시 리스트인 temp에 추가
        run2item_features[run] = np.stack(temp) #생성된 모든 아이템의 특징을 numpy배열로 변환하여 딕셔너리에 저장       Q) np.array(temp)와 뭐가 다른거지? -> np.stack은 np.concatenate과 유사하게 axis설정하여 여러 array 합침. 이 경우는 동일

        run2item_values[run] = np.ones((training_config['num_items'],)) #모든 아이템의 가치가 1로 초기화된 배열 저장(단순화하기위함,모든 아이템이 동일한 가치를 갖는다고 가정)

    return run2item_features, run2item_values

def set_model_params(rng, CTR_mode, winrate_mode, context_dim, feature_dim):
    #rng와 실험설정에 따라 CTR모델과 winrate모델의 파라미터를 생성하는 함수
    if CTR_mode=='bilinear':
    #ctr mode가 bilinear인 경우 context와 feature사이의 이중 선형 모델을 사용
        CTR_param = rng.normal(0.0, 1.0, size=(context_dim, feature_dim))
        #context_dim과 feature_dim 크기의 랜덤한 값을 가지는 2D배열 CTR_param을 생성
    elif CTR_mode=='MLP': #다층 퍼셉트론 모델을 사용
        d = context_dim * feature_dim #입력과 출력의 크기
        w1 = rng.normal(0.0, 1.0, size=(d, d)) #가중치 w1
        b1 = rng.normal(0.0, 1.0, size=(d, 1)) #편향 b1
        w2 = rng.normal(0.0, 1.0, size=(d, 1)) #가중치 w2
        b2 = rng.normal(0.0, 1.0, size=(1, 1)) #편향 b2
        CTR_param = (w1, b1, w2, b2) #튜플
    else:
        raise NotImplementedError
    
    if winrate_mode=='simulation':
        winrate_param = []
        for _ in range(2): #두 개의 모델 파라미터 생성
            temp = []
            for _ in range(training_config['num_items']):
                feature = rng.normal(0.0, 1.0, size=feature_dim)
                temp.append(feature)
            winrate_param.append(np.stack(temp))
    else:
        raise NotImplementedError #winrate model이 simulation이 아닌 경우는 처리할 수 없음

    return CTR_param, winrate_param

def instantiate_agent(rng, name, item_features, item_values, context_dim, buffer, agent_config):
    #주어진 파라미터와 설정에 따라 에이전트를 생성하여 반환하는 함수
    #Bandit, DQN, TD3중 하나의 에이전트를 선택하고 초기화

    #어떤 종류의 에이전트를 생성할지 결정
    if agent_config['type']=='Bandit':
    #type이 bandit인 경우, Bandit:무작위로 아이템을 선택하는 간단한 기준으로 행동하는 에이전트
        return Bandit(rng, name, item_features, item_values, context_dim, buffer, agent_config)
    elif agent_config['type']=='DQN':
    #type이 DQN인 경우, DQN: Dueling Deep Q-Network
        return DQN(rng, name, item_features, item_values, context_dim, buffer, agent_config)
    elif agent_config['type']=='TD3':
    #type이 TD3인 경우, TD3: Twin Delayed Deep Deterministic Policy Gradient 연속적인 행동을 결정하는 에이전트
        return TD3(rng, name, item_features, item_values, context_dim, buffer, agent_config)

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser() #명령행 인수 정의하고 파싱
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--cuda', type=str, default='0')
    args = parser.parse_args()

    with open('config/training_config.json') as f:
        training_config = json.load(f)
    
    with open(args.config) as f:
        agent_config = json.load(f)

    # Set up Random Generator
    # 난수 생성에 사용되는 랜덤 시드를 설정하는 과정(실험의 재현성을 위해 항상 동일한 시드로 난수를 생성)
    # 시드값이 동일하면 동일한 난수 생성됨
    rng = np.random.default_rng(training_config['random_seed'])
    np.random.seed(training_config['random_seed'])

    # training loop config
    num_runs = training_config['num_runs']
    num_episodes  = training_config['num_episodes']
    record_interval = training_config['record_interval']
    update_interval = training_config['update_interval']
    horizon = training_config['horizon']
    budget = training_config['budget']

    # context, item feature config
    context_dim = training_config['context_dim']
    feature_dim = training_config['feature_dim']
    context_dist = training_config['context_distribution']

    # CTR, winrate model
    CTR_mode = training_config['CTR_mode']
    winrate_mode = training_config['winrate_mode']

    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    print("running in {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    # Parse configuration file
    output_dir = agent_config['output_dir']
    output_dir = output_dir + '/' + time.strftime('%y%m%d-%H%M%S') + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # config file copy
    shutil.copy(args.config, os.path.join(output_dir, 'agent_config.json'))
    shutil.copy('config/training_config.json', os.path.join(output_dir, 'training_config.json'))

    # memory recording statistics
    reward = np.zeros((num_runs, num_episodes))
    win_rate = np.zeros((num_runs, num_episodes))
    optimal_selection = np.zeros((num_runs, num_episodes))
    episode_length = np.zeros((num_runs, num_episodes))
    uncertainty = np.zeros((num_runs, num_episodes))
    budget_left = np.zeros((num_runs, num_episodes))
    CTR_error = np.zeros((num_runs, num_episodes))
    winrate_error = np.zeros((num_runs, num_episodes))
    budgets = []
    bids = []

    run2item_features, run2item_values = draw_features(rng, num_runs, feature_dim)

    for run in range(num_runs):
        item_features = run2item_features[run]
        item_values = run2item_values[run] #1로 초기화된 상태

        #CTR모델과 winrate모델의 파라미터를 설정
        CTR_param, winrate_param = set_model_params(rng, CTR_mode, winrate_mode, context_dim, feature_dim)
        #CTR 모델 초기화
        CTR_model = CTR(CTR_mode, context_dim, item_features, CTR_param)
        if winrate_mode!='simulation': #winrate 모드가 simulation이 아니면 초기화
            winrate_model = Winrate(winrate_mode, context_dim, winrate_param)
        else:# winrate모드가 simulation이면(실제 데이터를 사용할 수 없는 상황)기존 CTR모델 사용하여 근사적으로 계산
            winrate_model = Winrate(winrate_mode, context_dim, winrate_param, CTR_model)

        buffer = Buffer()

        #에이전트 초기화
        agent = instantiate_agent(rng, agent_config['name'], item_features, item_values, context_dim, buffer, agent_config)
        auction = Auction(rng, agent, CTR_model, winrate_model, item_features, item_values, context_dim, context_dist, horizon, budget)
        # 에이전트 객체의 auction속성을 auction객체로 설정
        agent.auction = auction
        try:
            # agent.allocator가 OracleAllocator클래스를 상속받은 경우에만 CTR모델의 파라미터를 설정
            if isinstance(agent.allocator, OracleAllocator):
                agent.allocator.set_CTR_model(auction.CTR_model.model.M)
        except:
            pass
        
        # probw = np.zeros((100,10))
        # for k in range(100):
        #     context = auction.generate_context()
        #     bidding = np.linspace(0.1, 1.0, 10)
        #     prob_win = auction.winrate_model(context, bidding)
        #     probw[k] = np.array(prob_win).reshape(-1)

        # df_rows = {'context': [], 'bidding': [], 'probw': []}
        # for k in range(100):
        #     for l in range(10):
        #         df_rows['context'].append(k)
        #         df_rows['bidding'].append(bidding[l])
        #         df_rows['probw'].append(probw[k,l])
        # q_df = pd.DataFrame(df_rows)

        # fig, axes = plt.subplots(figsize=FIGSIZE)
        # plt.title('probw', fontsize=FONTSIZE + 2)
        # sns.lineplot(data=q_df, x="bidding", y='probw',hue='context', ax=axes)
        # plt.ylabel('prob', fontsize=FONTSIZE)
        # plt.xlabel("bidding", fontsize=FONTSIZE)
        # plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        # plt.tight_layout()
        # plt.savefig(f"{output_dir}/probw.png", bbox_inches='tight')
        # exit(1)

        t = 0 # 타임 스텝
        for i in tqdm(range(num_episodes), desc=f'run {run}'): #현재 run에서 설정한 num_episodes만큼 학습 루프 반복
            s, _ = auction.reset() #auction 환경 리셋하고 초기 상태 얻음
            done = truncated = False # 에피소드 종료 여부와 탐색기간 종료 여부 초기화
            start = t #에피소드의 시작 타임스텝을 저장
            episode_reward = 0
            episode_win = 0
            episode_optimal_selection = 0
            agent.set_exploration_param(i)

            while not (done or truncated):
                
                item, bidding = agent.bid(s, t) #현재상태 s에서 아이템과 입찰가 결정
                a = {'item' : item, 'bid' : np.array([bidding])} #환경에 적용할 행동을 딕셔너리 형태로 정의
                s_, r, done, truncated, info = auction.step(a) #환경에 행동을 적용하고 다음상태, 보상, 에피소드 종료 여부
                #현재상태, 선택한 아이템, 입찰가, 보상, 다음 상태를 버퍼에 저장
                buffer.append(s, item, bidding, r, s_, (done or truncated), info['win'], info['outcome'])
                t += 1
                episode_reward += r # 각 에피소드에서 얻은 누적 보상
                episode_win += float(info['win']) # 성공적으로 입찰에 성공한 횟수
                episode_optimal_selection += float(info['optimal_selection']) #최적의 아이템을 선택한 횟수
                s = s_ #상태를 다음 상태로 업데이트
                # 각 에피소드에서 선택한 선택한 입찰가와 예산을 기록
                budgets.append(s[-2]) 
                bids.append(bidding)
                #일정 주기로 학습을 진행하고 버퍼에 저장된 정보를 이용하여 에이전트의 학습을 수행하고 policy를 개선
                if t%update_interval==0:
                    agent.update(int(t/update_interval))

            #각 에피소드 정보 기록
            reward[run,i] = episode_reward #run번째 실험에서 i번째 에피소드의 누적 보상 기록 (에이전트가 해당 에피소드에서 얻은 총 보상)
            win_rate[run,i] = episode_win / (t - start) #run번째 실험에서 i번째 에피소드의 입찰 성공률을 기록(에이전트가 해당 에피소드에서 입찰에 성공한 횟수를 전체 시도횟수로 나눈 비율)
            optimal_selection[run,i] = episode_optimal_selection / (t - start) #run번째 실험에서 i번째 에피소드의 최적 아이템 선택 비율을 기록(에이전트가 해당 에피소드에서 최적 아이템을 선택한 횟수를 전체 시도횟수로 나눈 비율)
            episode_length[run,i] = t - start #run번째 실험에서 i번째 에피소드의 길이를 기록(해당 에피소드가 몇개의 타임 스템으로 구성되었는가)
            uncertainty[run,i] = agent.get_uncertainty(t-start) #run번째 실험에서 i번째 에피소드의 불확실성
            budget_left[run,i] = s[-2] #run번째 실험에서 i번째 에피소드의 남은 예산(에피소드가 해당 에피소드에서 마지막 타임 스텝에서 남은 예산)

            context = auction.generate_context()
            try:
                estim_ctr = agent.allocator.estimate_CTR(context) #에이전트의 CTR모델을 통해 예측한 값 반환
                true_ctr = auction.CTR_model(context) #실제 CTR값을 얻는 함수
                CTR_error[run,i] = np.mean((estim_ctr-true_ctr)**2) # 위 두 값의 차이를 제곱하고 평균을 구하여 CTR에러를 계산
            except: #예측 모델에 따라 estim_ctr 또는 estim_winrate를 얻을 수 없는 경우 에러 발생(예외 처리)
                CTR_error[run,i] = 0.0
            try:
                b_grid = np.linspace(0.1, 1, 10).reshape(-1,1)
                x = torch.Tensor(np.concatenate([np.tile(context, (10,1)), b_grid], axis=1)).to(agent.device)
                estim_winrate = agent.winrate.winrate_model(x).numpy(force=True).reshape(-1)#에이전트의 winrate 모델을 통해 예측한 값
                true_winrate = []
                for j in range(10):
                    true_winrate.append(auction.winrate_model(context, b_grid[j])) #실제 winrate값을 얻는 함수
                true_winrate = np.array(true_winrate)
                winrate_error[run,i] = np.mean((estim_winrate-true_winrate)**2) #run번째 실험에서 i번째 에피소드의 winrate모델 에러
            except:
                winrate_error[run,i] = 0.0
            
    flag = True # plot q-value shape if true
    states, item_inds, biddings, rewards, next_states, dones = agent.buffer.sample(1000)

    #Q-값을 시각화하기 위한 격자점을 생성
    #budget과 horizon을 각각 50개와 20개의 점으로 나누어 격자점을 생성 -> 2D배열 생성
    temp = np.concatenate([np.tile(np.linspace(0,budget,50),(20,1)).reshape(-1,1),
                           np.tile(np.linspace(0, horizon, 20),(1,50)).reshape(-1,1)], axis=1)
    contexts = states[:, :context_dim] #주어진 states에서 컨텍스트 정보(에이전트의 의사 결정에 영향을 주는 정보)를 추출
    q = np.zeros((1000,3)) # Q-값을 저장할 2D배열을 초기화
    q[:,:-1] = temp 
    biddings = np.random.uniform(0,1,size=1000) #무작위로 0과 1사이의 값으로 채워진 1D배열을 생성

    #알고리즘이 TD3 인지 DQN인지에 따라 다른 방식으로 Q-값을 계산
    if isinstance(agent, TD3):
        # TD3인 경우 contexts, temp, biddings를 하나의 텐서로 합쳐서 x를 생성
        x = torch.Tensor(np.concatenate([contexts, temp, biddings.reshape(-1,1)], axis=1)).to(agent.device)
        q[:,-1] = agent.critic.Q1(x).numpy(force=True).reshape(-1)# 마지막 열에 저장
    elif isinstance(agent, DQN):
        # DQN인 경우 contexts, agent.items[iten_inds], temp, biddings를 하나의 텐서로 합쳐서 x를 생성
        x = torch.Tensor(np.concatenate([contexts, agent.items[item_inds], temp, biddings.reshape(-1,1)], axis=1)).to(agent.device)
        q[:,-1] = agent.local_network.Q1(x).numpy(force=True).reshape(-1) #마지막 열에 저장
    else:
        flag = False
    # flag가 False이면 시각화 하지 않음

    if flag:
        df_rows = {'Step Left': [], 'Budget Left': [], 'Q': []}
        for i in range(q.shape[0]):
            df_rows['Budget Left'].append(q[i,0])
            df_rows['Step Left'].append(q[i,1])
            df_rows['Q'].append(q[i,2])
        q_df = pd.DataFrame(df_rows)

        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title('Avg Q value', fontsize=FONTSIZE + 2)
        sns.lineplot(data=q_df, x="Step Left", y='Q', ax=axes)
        plt.ylabel('Q', fontsize=FONTSIZE)
        plt.xlabel("Step Left", fontsize=FONTSIZE)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/q_vs_step.png", bbox_inches='tight')

        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title('Avg Q value', fontsize=FONTSIZE + 2)
        sns.lineplot(data=q_df, x="Budget Left", y='Q', ax=axes)
        plt.ylabel('Q', fontsize=FONTSIZE)
        plt.xlabel("Budget Left", fontsize=FONTSIZE)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/q_vs_budget.png", bbox_inches='tight')

    budget_array = np.sort(np.array(budgets))
    df_rows = {'Encountered Budget': []}
    for i in range(budget_array.shape[0]):
        df_rows['Encountered Budget'].append(budget_array[i])
    budgets_df = pd.DataFrame(df_rows)
    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title('Encountered Budget', fontsize = FONTSIZE + 2)
    sns.histplot(data=budgets_df, x="Encountered Budget", bins=100)
    plt.savefig(f"{output_dir}/encountered_budgets.png", bbox_inches='tight')

    budget_array = np.sort(np.array(budgets[-1000:]))
    df_rows = {'Encountered Budget': []}
    for i in range(budget_array.shape[0]):
        df_rows['Encountered Budget'].append(budget_array[i])
    budgets_df = pd.DataFrame(df_rows)
    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title('Encountered Budget (last 1000steps)', fontsize = FONTSIZE + 2)
    sns.histplot(data=budgets_df, x="Encountered Budget", bins=100)
    plt.savefig(f"{output_dir}/encountered_budgets_last.png", bbox_inches='tight')

    bidding_array = np.sort(np.array(bids))
    df_rows = {'bidding': []}
    for i in range(bidding_array.shape[0]):
        df_rows['bidding'].append(bidding_array[i])
    bidding_df = pd.DataFrame(df_rows)
    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title('Bidding', fontsize = FONTSIZE + 2)
    sns.histplot(data=bidding_df, x="bidding", bins=10)
    plt.savefig(f"{output_dir}/biddings.png", bbox_inches='tight')

    bidding_array = np.sort(np.array(bids[-1000:]))
    df_rows = {'bidding': []}
    for i in range(bidding_array.shape[0]):
        df_rows['bidding'].append(bidding_array[i])
    bidding_df = pd.DataFrame(df_rows)
    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title('Bidding (last 1000steps)', fontsize = FONTSIZE + 2)
    sns.histplot(data=bidding_df, x="bidding", bins=10)
    plt.savefig(f"{output_dir}/biddings_last.png", bbox_inches='tight')

    reward = average(reward, record_interval)
    optimal_selection = average(optimal_selection, record_interval)
    prob_win = average(win_rate, record_interval)

    cumulative_reward = np.cumsum(reward, axis=1)

    reward_df = numpy2df(reward, 'Reward')
    reward_df.to_csv(output_dir + '/reward.csv', index=False)
    plot_measure(reward_df, 'Reward', record_interval, output_dir)

    cumulative_reward_df = numpy2df(cumulative_reward, 'Cumulative Reward')
    plot_measure(cumulative_reward_df, 'Cumulative Reward', record_interval, output_dir)

    optimal_selection_df = numpy2df(optimal_selection, 'Optimal Selection Rate')
    optimal_selection_df.to_csv(output_dir + '/optimal_selection_rate.csv', index=False)
    plot_measure(optimal_selection_df, 'Optimal Selection Rate', record_interval, output_dir)

    prob_win_df = numpy2df(prob_win, 'Probability of Winning')
    prob_win_df.to_csv(output_dir + '/prob_win.csv', index=False)
    plot_measure(prob_win_df, 'Probability of Winning', record_interval, output_dir)

    episode_length_df = numpy2df(episode_length, 'Episode Length')
    episode_length_df.to_csv(output_dir + '/episode_length.csv', index=False)
    plot_measure(episode_length_df, 'Episode Length', 1, output_dir)

    uncertainty_df = numpy2df(uncertainty, 'Uncertainty')
    uncertainty_df.to_csv(output_dir + '/uncertainty.csv', index=False)
    plot_measure(uncertainty_df, 'Uncertainty', 1, output_dir)

    budget_df = numpy2df(budget_left, 'Remaining Budget')
    budget_df.to_csv(output_dir + '/budget.csv', index=False)
    plot_measure(budget_df, 'Remaining Budget', 1, output_dir)

    CTR_error_df = numpy2df(CTR_error, 'CTR Error')
    plot_measure(CTR_error_df, 'CTR Error', 1, output_dir)

    winrate_error_df = numpy2df(winrate_error, 'Win Rate Error')
    plot_measure(winrate_error_df, 'Win Rate Error', 1, output_dir)