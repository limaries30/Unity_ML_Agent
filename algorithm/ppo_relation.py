import numpy as np
import random
from mlagents.envs import UnityEnvironment
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import time
from prey import PreyPPO
import torch.nn.functional as F

start=time.time()
summary = SummaryWriter('./logs/'+str(start))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def makeO(state,nums=2,numa=4): #S*A
    O=np.zeros((1,3,4))
    for idx,e in enumerate(state):
        O[0,:,idx]=state[idx]
        #O[0,[2,3],idx]=state[1][[0,1]]
    return O


class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        #input size :(2*D_s+D_r)
        super(RelationalModel, self).__init__()
        
        self.output_size = output_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)
        #x size :batch * n_relation,2D_s+D_r
        x = self.layers(x) 
        x = x.view(batch_size, n_relations, self.output_size)
        return x

class Actor(nn.Module):
    def __init__(self, input_size, action_size, hidden_size):
        #input size :(2*D_s+D_r)
        super(Actor, self).__init__()
        
        self.action_size = action_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
    
    def forward(self, x):
        '''
        Args:
            x: [batch_size, N_o, input_size]
        Returns:
            [batch_size,  N_o, action_size]
        '''
        batch_size,  N_o, input_size = x.size()
        x = x.view(-1, input_size)
        #x size :batch * n_relation,2D_s+D_r
        x = self.layers(x) 
        x = F.softmax(x.view(batch_size, N_o, self.action_size),dim=2)
        
        return x

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        #input size :(2*D_s+D_r)
        super(Critic, self).__init__()
        
        
        self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        batch_size,  N_o, input_size = x.size()
        x = x.view(-1, input_size)
        #x size :batch * n_relation,2D_s+D_r
        x = self.layers(x) 
        x = x.view(batch_size, N_o)
        #x=torch.mean(x,dim=1,keepdim=True)
        
        return x

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var,sender_relation,receiver_relation,relation_a,effect_size,state_r,relation_model):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = Actor(state_r+effect_size,action_dim,n_latent_var)
        
        # critic
        self.value_layer = Critic(state_r+effect_size, n_latent_var)

        self.relation_layer=relation_model
        #self.relation_layer=RelationalModel(2*state_r+1,effect_size,n_latent_var)

        self.sender_relations = torch.from_numpy(sender_relation).float()
        self.reciever_relations = torch.from_numpy(receiver_relation).float() #[1,N_o,N_r]
        self.relation_a =  torch.from_numpy(relation_a).float()
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        #state: [Batch,N_o,N_s
        state=torch.from_numpy(state).float()
        state_r = state.bmm(self.reciever_relations)
        state_s = state.bmm(self.sender_relations)
        B= torch.cat([state_r,state_s,self.relation_a],1).permute(0,2,1) #[Batch,N_r,2*D_s+1]
        
        effects= self.relation_layer(B) # output size :[batch,n_relation,effect_size]
        effect_receivers = self.reciever_relations.bmm(effects) #[batch,N_o,effect_size]
        C= torch.cat([state.permute(0,2,1),effect_receivers],2)   #[:,[0,2,3],:] #[Batch,N_0,N_s+effect_size]
        action_probs = self.action_layer(C)
        dist = Categorical(action_probs)
        action = dist.sample().squeeze(0)
        
        memory.states.append(C)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action
    
    def evaluate(self, state, action):
        state=state.squeeze(1)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip,sender_relation,receiver_relation,relation_a,effect_size,state_r,state_ra):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.relation_layer= RelationalModel(2*state_r+state_ra,effect_size,n_latent_var)
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var,sender_relation,receiver_relation,relation_a,effect_size,state_r,self.relation_layer).to(device)
        self.actor_optimizer = torch.optim.Adam(self.policy.action_layer.parameters(), lr=lr, betas=betas)
        self.critic_optimizer = torch.optim.Adam(self.policy.value_layer.parameters(), lr=lr, betas=betas)
        self.relation_optimizer= torch.optim.Adam(self.relation_layer.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var,sender_relation,receiver_relation,relation_a,effect_size,state_r,self.relation_layer).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        #globalreward
        #rewards = torch.tensor(rewards).unsqueeze(1).float()
        #rewards = torch.tensor(rewards)[:,[0,2,3]].float()
        rewards=torch.tensor(rewards).float()
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device)
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach().squeeze(1)
        batchsize=old_states.size()[0]
        minibatchsize=128
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            for i in range(int(batchsize/minibatchsize)):
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[i*minibatchsize:(i+1)*minibatchsize], old_actions[i*minibatchsize:(i+1)*minibatchsize])
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach()[i*minibatchsize:(i+1)*minibatchsize])
                    
                # Finding Surrogate Loss:
        
                rewardbatch = rewards[i*minibatchsize:(i+1)*minibatchsize]
                #rewardbatch = rewards[i*minibatchsize:(i+1)*minibatchsize].unsqueeze(1)
                # loss = self.MseLoss(torch.mean(state_values,dim=1,keepdim=True), rewardbat
                # ch) 

                advantages =rewardbatch - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 10*self.MseLoss(state_values, rewardbatch) - 0.01*dist_entropy
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.relation_optimizer.zero_grad()
                loss.mean().backward(retain_graph=True)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.relation_optimizer.step()


                # take gradient step
        
        # Copy new weights into old policy:
        self.policy_old.action_layer.load_state_dict(self.policy.action_layer.state_dict())
        self.policy_old.value_layer.load_state_dict(self.policy.value_layer.state_dict())
        

def main():
    ############## Hyperparameters ##############
    game = "Predator_Prey"
    env_name = "../env/" + game + "/Windows/" + game
    env = UnityEnvironment(file_name=env_name)

    # Brain setting 
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    num_agent=3
    env_config = {"Num_Agent":num_agent} #predator 수
    train_mode=True
    action_size=4
    
    state_dim = (num_agent+1)*3
    state_r = 3
    print('state dim',state_dim)
    action_dim = action_size**num_agent
    
    
    solved_reward = 230         # stop training if avg_reward > solved_reward
    
    log_interval = 30           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 512           # number of variables in hidden layer
    
    update_timestep =128    # update policy every n timesteps
    lr = 0.00001
    betas = (0.9, 0.999)
    gamma = 0.93             # discount factor
    K_epochs = 10                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    save_interval=100

    AGENT_NUM=num_agent+1
    Num_relation_state=2
    Num_relation=AGENT_NUM*(AGENT_NUM-1) # reciever이면서 동시에 sender이다

    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        #env.seed(random_seed)
    
    memory = Memory()
    preymemory = Memory()


    Rr_input = np.zeros([1, AGENT_NUM, Num_relation])
    Rs_input = np.zeros([1, AGENT_NUM, Num_relation])
    Ra_input = np.zeros([1, Num_relation_state, Num_relation])
    relation_idx = 0
    for i in range(AGENT_NUM):
        for j in range(AGENT_NUM):
            if i is not j:
                Rs_input[:,i,relation_idx] = 1.0
                Rr_input[:,j,relation_idx] = 1.0
                if i==1:
                    Ra_input[:,0,relation_idx]=1
                if j==1:
                    Ra_input[:,1,relation_idx]=1
                relation_idx = relation_idx + 1
    effect_size = 128
    relation_net = RelationalModel(2*state_r+1,effect_size,n_latent_var)
    ppo = PPO(state_dim, action_size, n_latent_var, lr, betas, gamma, K_epochs,eps_clip,Rs_input,Rr_input,Ra_input,effect_size,state_r,Num_relation_state)
    # prey_ppo  = PreyPPO(state_dim, action_size, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    # prey_name ='prey_latent_var_512_lr_00002'
    # prey_ppo.policy.load_state_dict(torch.load('./model/'+prey_name))
    # prey_ppo.policy_old.load_state_dict(torch.load('./model/'+prey_name))
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    env_info = env.reset(train_mode=train_mode,config=env_config)[brain_name]

    keep_train=False

    if keep_train or not train_mode: 
        #name='latent_var_'+str(n_latent_var)+'_lr_'+str(lr)+'_epi_'+'1400'
        name='latent_var_512_lr_1e-06_epi_50000'
        ppo.policy.action_layer.load_state_dict(torch.load('./model/relation_ppo/actor_'+name))
        ppo.policy_old.action_layer.load_state_dict(torch.load('./model/relation_ppo/actor_'+name))
        ppo.policy.value_layer.load_state_dict(torch.load('./model/relation_ppo/critic_'+name))
        ppo.policy_old.value_layer.load_state_dict(torch.load('./model/relation_ppo/critic_'+name))
        ppo.policy.relation_layer.load_state_dict(torch.load('./model/relation_ppo/relation_'+name))
        ppo.policy_old.relation_layer.load_state_dict(torch.load('./model/relation_ppo/relation_'+name))
        print('loaded')

    predator_score = 0
    prey_score=0


    # training loop
    for i_episode in range(1, max_episodes+1):
        
        score = np.zeros([num_agent])
        env_info = env.reset(train_mode=train_mode)[brain_name]
        #state = np.array(env_info.vector_observations).reshape(-1)
        O = makeO(np.array(env_info.vector_observations))
        #state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            if not train_mode:
                time.sleep(0.1)
     
            
            # Running policy_old:
            action = ppo.policy_old.act(O, memory).view(-1)
            action_list=[]
            rdx=0
            
            #동시에 action
            # for i in range(num_agent+1):
            #     action_list.append(action[i].item())
                # if i==1:
                #     # prey_act = prey_ppo.policy_old.act(state,preymemory)
                #     # action_list.append(int(prey_act))
                #     prey_act=random.randint(0, action_size)
                #     action_list.append(prey_act)
                # else:
                #     action_list.append(action[rdx].item())
            
                #     rdx+=1
            # prey 먼저   
            prey_actionlist=[4]*(num_agent+1)
            predator_actionlist=[4]*(num_agent+1)
            prey_action = action[1].item()
            for i in range(num_agent+1):
                if i!=1:
                    predator_actionlist[i]=action[i].item()

            ##prey 먼저 act
            prey_actionlist[1]=prey_action
            env_info = env.step(prey_actionlist)[brain_name]
            ##prey reward
            rewards1 = env_info.rewards

            terminals1 = env_info.local_done
            if True in terminals1:
                O = makeO(np.array(env_info.vector_observations))
                reward = np.array(rewards1)
                reward_predator=reward[[0,2,3]].sum()
                reward_prey=reward[1]
                # if rewards[np.argmin(role)] < -0.7:
                #     for i in range(len(rewards)):
                #         if role[i] == 1 and rewards[i] < 0.7:
                #             rewards[i] = 0.3
            
                done = np.array(env_info.local_done)[1]
                # Saving reward and is_terminal:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                preymemory.rewards.append(reward_prey)
                preymemory.is_terminals.append(done)
                
                # update if its time
                if timestep % update_timestep == 0 and train_mode:
                    ppo.update(memory)
                    memory.clear_memory()
                    #prey_ppo.update(preymemory)
                    #preymemory.clear_memory()
                    timestep = 0

                # if not train_mode:
                #     time.sleep(0.1)
                
                running_reward += reward_predator

                predator_score += reward_predator
                prey_score += reward_prey
                break

            #predator action
            env_info = env.step(predator_actionlist)[brain_name]

            # 다음 상태, 보상, 게임 종료 정보 취득
            #  
            O = makeO(np.array(env_info.vector_observations))
            rewards2 = env_info.rewards
            reward = np.array([a + b for a, b in zip(rewards1, rewards2)])
            reward_predator=reward[[0,2,3]].sum()
            reward_prey=reward[1]
            # if rewards[np.argmin(role)] < -0.7:
            #     for i in range(len(rewards)):
            #         if role[i] == 1 and rewards[i] < 0.7:
            #             rewards[i] = 0.3
        
            done = np.array(env_info.local_done)[1]
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            preymemory.rewards.append(reward_prey)
            preymemory.is_terminals.append(done)
            
            # update if its time
            if timestep % update_timestep == 0 and train_mode:
                ppo.update(memory)
                memory.clear_memory()
                #prey_ppo.update(preymemory)
                #preymemory.clear_memory()
                timestep = 0

            # if not train_mode:
            #     time.sleep(0.1)
            
            running_reward += reward_predator

            predator_score += reward_predator
            prey_score += reward_prey

            if done:
                break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
        #     break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            predator_score = predator_score/log_interval
            prey_score = prey_score/log_interval

            summary.add_scalar('pred_score',predator_score, i_episode)
            summary.add_scalar('prey_score',prey_score, i_episode)
            summary.add_scalar('epi_steps',avg_length, i_episode)
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, predator_score))
            predator_score = 0
            predator_score=0
            avg_length = 0

        if i_episode % save_interval == 0:
            name='latent_var_'+str(n_latent_var)+'_lr_'+str(lr)+'_epi_'+str(i_episode)
            torch.save(ppo.policy.action_layer.state_dict(), './model/relation_ppo_keeptrain/actor_'+name)
            torch.save(ppo.policy.value_layer.state_dict(), './model/relation_ppo_keeptrain/critic_'+name)
            torch.save(ppo.policy.relation_layer.state_dict(), './model/relation_ppo_keeptrain/relation_'+name)

if __name__=="__main__":
    main()