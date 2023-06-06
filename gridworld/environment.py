import numpy as np

class WindyGridworld():
    up = [0, 1]
    down = [0, -1]
    right = [1, 0]
    left = [-1, 0]

    #action space
    A = np.array([up, down, right, left])

    def __init__(self, num_concepts, method, knn, means, policy_model) -> None:
        self.num_concepts = num_concepts
        self.method = method #simple or knn to dictate how clusters are assigned
        self.knn = knn #A knn predictor already trained
        self.means = means
        self.policy_model = policy_model

    def _policy(self, s):
        if self.policy_model is None:
            indices = [0, 1, 2, 3]
            # higher probability of moving up and right
            return self.A[np.random.choice(indices, 1, p=[0.35, 0.15, 0.35, 0.15])[0]]
        else:
            s = s.reshape((1,2))
            action_index = np.argmax(self.policy_model.predict(s)) #predict actions using pi_b
            return self.A[action_index] 


    def _reached_goal(self, s):
        if s.shape[0] == 1:
            s = s[0]
        return s[0] >= 3 and s[0] <= 4 and s[1] >= 3 and s[1] <= 4


    def _wind_simple(self, s):
        # Different clusters experience different levels of wind
        severities = [0.1, 0.3, 0.5, 0.7, 0.9]  # Wind severity per cluster
        distances = []
        for i in range(self.num_concepts):
            distances.append(np.linalg.norm(s - self.means[i]))
        assigned_cluster = np.argmin(distances)
        return severities[assigned_cluster]*np.array([-1, -1]), assigned_cluster

    def _wind_knn(self, s):
    
        severities = [0.1, 0.3, 0.5, 0.7, 0.9]  # Wind severity per cluster
        assigned_cluster = self.knn.predict(s.reshape(1,2))[0]
    
        return severities[assigned_cluster]*np.array([-1, -1]), assigned_cluster


    def _wind(self, s):
        if self.method == "knn":
            return self._wind_knn(s)
        elif self.method == "simple":
            return self._wind_simple(s)
        else:
            return self._wind_simple(s)
    

    def play(self, saveOnGoalReached=False, trajectories=[], rewards=[], unique_concepts=[], concepts=[], actions=[], s = np.array([-3, -3])):
        T = 1000 #Maximum episode length
        alpha = 0.08
        beta = 0.01
        history = []
        concept_history = []
        action_history = []
        reward = 0
        visited_concepts = []
        for _ in range(1, T):
            
            action = self._policy(s)
            wind_value, concept = self._wind(s)
            s = s + alpha*action + beta*wind_value  # transition

            history.append(s)
            concept_history.append(concept)
            action_history.append(action)
            if concept not in visited_concepts:
                visited_concepts.append(concept)
            if self._reached_goal(s):
                if saveOnGoalReached:
                    trajectories.append(np.array(history))
                    concepts.append(np.array(concept_history))
                    actions.append(np.array(action_history))
                    rewards.append(np.array(reward))
                    visited_concepts.sort()
                    unique_concepts.append(visited_concepts)
                break
            reward -= 1

        history = np.array(history)
        return history


    def save_dataset_and_trajectories(self, filename, centroids, clusters, trajectories, rewards, unique_concepts, concepts, actions):
        np.savez(filename, centroids=centroids, clusters=clusters, trajectories=np.array(trajectories,
                dtype=object), rewards=np.array(rewards), unique_concepts=np.array(unique_concepts, dtype=object), concepts=np.array(concepts, dtype=object), actions=np.array(actions, dtype=object))


    def load_dataset_and_trajectories(self, filename):
        npz = np.load(filename, allow_pickle=True)
        print(npz.files)
        return npz['centroids'], npz['clusters'], npz['trajectories'], npz['rewards'], npz['unique_concepts'], npz['concepts'], npz['actions']
        
