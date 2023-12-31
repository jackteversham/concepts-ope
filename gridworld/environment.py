import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class WindyGridworld:
    up = [0, 1]
    down = [0, -1]
    right = [1, 0]
    left = [-1, 0]

    # action space
    A = np.array([up, down, right, left])

    def __init__(
        self,
        num_concepts,
        method,
        knn,
        means,
        policy_model=None,
        custom_eval_policy=None,
        concept_model=None,
        *args,
        **kwargs,
    ) -> None:
        self.num_concepts = num_concepts
        self.method = method  # simple or knn to dictate how clusters are assigned
        self.knn = knn  # A knn predictor already trained
        self.means = means
        self.policy_model = policy_model
        self.custom_eval_policy = custom_eval_policy
        self.concept_model = concept_model
        self.args = args
        self.kwargs = kwargs

    def _policy(self, s, history, distribution):
        if self.custom_eval_policy is not None:
            #must reurn an action and a distribution
            return self.custom_eval_policy(s, self.A, history, self.concept_model, self.policy_model, self.kwargs)

        elif self.policy_model is not None:
            s = s.reshape((1, 2))
            if self.concept_model is not None:
                s = self.concept_model(s)  # predict on concept instead

            distribution = self.policy_model.predict(s)[0]
            argmax = np.argmax(distribution)
            epsilon = 0.9
            option = np.random.choice([0, 1], 1, p=[1 - epsilon, epsilon])[0]
            if option == 0:
                return self.A[argmax]  # return argmax with probability 1-eps
            # action_index = np.argmax(distribution) #predict actions using pi_b using a greedy strategy
            # NOTE: Consider here epsilon greedy: take argmax with probablilty epsilon, otherwise sample from the distribution
            indices = [0, 1, 2, 3]  # up, down, right, left
            return self.A[np.random.choice(indices, 1, p=distribution)[0]], distribution
       
        else:
            return self._default_behaviour_policy(s, distribution)

    def _eval_policy(self, s, history, distribution):
        if self.policy_model is not None:
            s = s.reshape((1, 2))
            if self.concept_model is not None:
                s = self.concept_model(s)  # predict on concept instead

            epsilon = self.kwargs["eval_epsilon"]
            distribution = self.policy_model.predict(s)[0]
            if s[0][0] < -1 and s[0][1] > 1.5:
                option = np.random.choice([0, 1], 1, p=[1 - epsilon, epsilon])[0]
                if option == 1:
                    distribution = self.kwargs["eval_distribution"]

            indices = [0, 1, 2, 3]  # up, down, right, left
            return self.A[np.random.choice(indices, 1, p=distribution)[0]], distribution

        elif self.custom_eval_policy is not None:
            return self.custom_eval_policy(
                s, self.A, history, self.concept_model, *self.args
            )

        else:
            return self._default_behaviour_policy(s, distribution)

    def _default_behaviour_policy(
        self, s, distribution
    ):  # the policy used to generate the dataset
        epsilon = self.kwargs["epsilon"]
        dist_index = np.random.choice([0, 1], 1, p=[1 - epsilon, epsilon])[0]
        # choose "optimal policy" or random policy in epsilon greedy fashion.
        # Then how to define optimal policy conditioned on state? Can I adjust epsilon based on concepts? or wind?
        # Or, adjust optimal policy based on region in state space (with no knowledge of concept)
        if self.num_concepts==5:
            optimal_distribution = self._optimal_policy_by_region(s)
        else:
            optimal_distribution = self._optimal_policy_by_region_more_concepts(s)
        distributions = [optimal_distribution, [0.25, 0.25, 0.25, 0.25]]
        p = distributions[dist_index]
        indices = [0, 1, 2, 3]  # up, down, right, left
        return self.A[np.random.choice(indices, 1, p=p)[0]], p

    def _argmax_behaviour_policy(self, s):
        epsilon = self.kwargs["epsilon"]
        dist_index = np.random.choice([0, 1], 1, p=[1 - epsilon, epsilon])[
            0
        ]  # choose "optimal policy" or random policy in epsilon greedy fashion.
        # Then how to define optimal policy conditioned on state? Can I adjust epsilon based on concepts? or wind?
        # Or, adjust optimal policy based on region in state space (with no knowledge of concept)
        optimal_distribution = self._optimal_policy_by_region(s)
        if dist_index == 0:
            return (
                self.A[np.argmax(optimal_distribution)],
                optimal_distribution,
            )  # should be returning [1,0,0,0] according to action_index here?
        else:
            indices = [0, 1, 2, 3]  # up, down, right, left
            return (
                self.A[np.random.choice(indices, 1, p=optimal_distribution)[0]],
                optimal_distribution,
            )

    def _optimal_policy_by_region(self, s):  # original dataset generation
        x, y = s[0], s[1]
        if x < 1 and y < 1:  # origin 1,1
            return [0.7, 0, 0.3, 0]  # bottom left
        elif x > 1 and y < 1:
            return [0.75, 0, 0.25, 0]  # bottom right
        elif x > 1 and y > 1:
            return [0.5, 0, 0.5, 0]  # top right
        else:
            return [0.2, 0.2, 0.6, 0]  # top left.

    def _optimal_policy_by_region_2(
        self, s, distribution
    ):  # This was added for the second round of experiments to try get more trajectories through concept four
        x, y = s[0], s[1]
        if x < 1 and y < 1:  # origin 1,1
            if x < -1 and y < -1:
                return distribution
            return [0.7, 0, 0.3, 0]  # bottom left
        elif x > 1 and y < 1:
            return [0.75, 0, 0.25, 0]  # bottom right
        elif x > 1 and y > 1:
            return [0.5, 0, 0.5, 0]  # top right
        else:
            return [0.25, 0.15, 0.6, 0]  # top left.

    def _optimal_policy_by_region_more_concepts(self, s):  # original dataset generation
        x, y = s[0], s[1]
        if x < 1 and y < 1:  # origin 1,1
            return [0.7, 0, 0.3, 0]  # bottom left
        elif x > 1 and y < 1:
            return [0.55, 0, 0.35, 0.1]  # bottom right
        elif x > 1 and y > 1:
            return [0.5, 0, 0.5, 0]  # top right
        else:
            return [0.2, 0.2, 0.6, 0]  # top left.

    def _reached_goal(self, s):
        if s.shape[0] == 1:
            s = s[0]
        return s[0] >= 3 and s[0] <= 4 and s[1] >= 3 and s[1] <= 4

    def _wind_simple(self, s):
        # Wind is defined only by state and position of dataset random concept centers. 
        # These same wind regions can be used for evaluating new concepts, and should be since 
        # this is how the dataset was generated
        # Different clusters experience different levels of wind
        severities = [0.2, 0.6, 1, 1.4, 1.8]  # Wind severity per cluster
        if self.num_concepts == 10:
            severities = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2.0]
        distances = []
        for i in range(self.num_concepts):
            distances.append(np.linalg.norm(s - self.means[i]))
        assigned_cluster = np.argmin(distances)
        return severities[assigned_cluster] * np.array([-1, -1]), assigned_cluster
    
    def _wind_knn(self, s):
        severities = [
            0.2,
            0.4,
            0.6,
            0.8,
            1,
            1.2,
            1.4,
            1.6,
            1.8,
            2.0,
        ]  # Wind severity per cluster (10 concepts)
        assigned_cluster = self.knn.predict(s.reshape(1, 2))[0]

        return severities[assigned_cluster] * np.array([-1, -1]), assigned_cluster

    def _wind(self, s):
        if self.method == "knn":
            return self._wind_knn(s)
        elif self.method == "simple":
            return self._wind_simple(s)
        else:
            return self._wind_simple(s)

    def play(
        self,
        saveOnGoalReached=False,
        trajectories=[],
        rewards=[],
        unique_concepts=[],
        concepts=[],
        actions=[],
        s=np.array([-3, -3]),
        use_eval_policy=False,
        distribution=[0.6, 0, 0.4, 0],  # for bottom left < (-1,-1)
    ):
        T = 600  # Maximum trajectory/episode length
        alpha = 0.08
        beta = 0.01
        history = []
        concept_history = []
        action_history = []
        reward = 0
        visited_concepts = []
        for _ in range(1, T):
            policy = self._eval_policy if use_eval_policy else self._policy
            action, _ = policy(s, history, distribution)
            wind_value, concept = self._wind(s)
            s = s + alpha * action + beta * wind_value  # transition

            history.append(s)  # append current state to history
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

    def save_dataset_and_trajectories(
        self,
        filename,
        centroids,
        clusters,
        trajectories,
        rewards,
        unique_concepts,
        concepts,
        actions,
    ):
        np.savez(
            filename,
            centroids=centroids,
            clusters=clusters,
            trajectories=np.array(trajectories, dtype=object),
            rewards=np.array(rewards),
            unique_concepts=np.array(unique_concepts, dtype=object),
            concepts=np.array(concepts, dtype=object),
            actions=np.array(actions, dtype=object),
        )

    def load_dataset_and_trajectories(self, filename):
        npz = np.load(filename, allow_pickle=True)
        print(npz.files)
        return (
            npz["centroids"],
            npz["clusters"],
            npz["trajectories"],
            npz["rewards"],
            npz["unique_concepts"],
            npz["concepts"],
            npz["actions"],
        )

    def plot_trajectory(self, history, means, clusters):
        fig, ax = plt.subplots(figsize=(16, 8))
        plt.plot(history[:, 0], history[:, 1])
        rect = patches.Rectangle([3, 3], 1, 1, fill=True, color="grey", alpha=0.5)
        ax.add_patch(rect)

        for i in range(len(means)):
            start = i * 100
            end = (i + 1) * 100
            plt.scatter(
                clusters[start:end, 0], clusters[start:end, 1], label=f"Cluster {i}"
            )

        plt.legend()
        plt.title("Trajectory through Windy Gridworld")
        plt.show()
