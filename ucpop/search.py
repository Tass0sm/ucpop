import heapq
import logging
import networkx as nx

logger = logging.getLogger(__name__)


def best_first_search(initial_state, daughters_fn, goal_p, rank_fn, limit):
    """perform best first search starting from the initial state in the graph
    induced by initial_state and daughters_fn, terminating according to goal_p,
    and ranking nodes to explore according to rank_fn. limit is a positive limit
    on the number of branches the search is allowed to encounter before
    terminating.
    """

    # priority queue for next plans to explore, along with set for faster
    # membership testing
    search_queue = [(0, initial_state)]
    search_queue_set = set([initial_state])

    # tree recording parent relationships
    search_tree = {}

    # set for plans already explored
    closed_set = set()

    # continue while there are still nodes to search and the limit hasn't been
    # exhausted.
    while search_queue and limit > 0:

        # get the next best node to explore
        _, current = heapq.heappop(search_queue)
        search_queue_set -= {current}

        logger.info(f"Exploring: {current}")
        # try:
        #     logger.info(f"Plan: {current.plan.to_partial_order_plan()}")
        # except nx.NetworkXUnfeasible:
        #     print("NODE HAS A CYCLE!")
        logger.info(f"Threats: {current.threats}")

        # if its the goal, end the search and return it
        if goal_p(current):
            return current, search_tree

        # otherwise mark it as having been explored
        closed_set.add(current)

        # get all the children from the current plan and compute their ranks
        daughters_and_extras, common_extras = daughters_fn(current)
        if len(daughters_and_extras) == 0:
            daughters, extras = [], []
        else:
            daughters, extras = list(zip(*daughters_and_extras))
        ranked_children = list(map(lambda c, es: (rank_fn(c), c, es), daughters, extras))
        num_children = len(ranked_children)

        # decrement limit by the number of children found here
        limit -= num_children

        # for each child, add it to the queue as long as we haven't already
        # explored it or we haven't already added it to the queue
        for rank, child, extras in ranked_children:
            if child in closed_set or child in search_queue_set:
                continue
            else:
                search_tree[child] = (current, extras | common_extras)
                heapq.heappush(search_queue, (rank, child))
                search_queue_set |= {child}

    print("Search Terminated")
    return None, search_tree


def depth_first_search(initial_state, daughters_fn, goal_p, rank_fn, limit):
    """perform depth first search starting from the initial state in the graph
    induced by initial_state and daughters_fn, terminating according to goal_p,
    and ranking nodes to explore first according to rank_fn. limit is a positive
    limit on the depth the search is allowed to reach before terminating.
    """

    # stack for plans to explore
    search_stack = [initial_state]
    search_queue_set = set([initial_state])

    # set for plans already explored
    closed_set = set()

    # continue while there are still nodes to search and the limit hasn't been
    # exhausted.
    while search_stack and limit > 0:

        # get the next best node to explore
        current = search_stack.pop()
        search_queue_set -= {current}

        logger.info(f"Exploring: {current}")
        try:
            logger.info(f"Plan: {current.plan.to_partial_order_plan()}")
        except nx.NetworkXUnfeasible:
            print("NODE HAS A CYCLE!")
        logger.info(f"Threats: {current.threats}")

        # if its the goal, end the search and return it
        if goal_p(current):
            return current

        # otherwise mark it as having been explored
        closed_set.add(current)

        # get all the children from the current plan and compute their ranks
        ranked_children = list(map(lambda c: (rank_fn(c), c), daughters_fn(current)))
        sorted_children = sorted(ranked_children, key=lambda x: x[0])
        num_children = len(sorted_children)

        # for each child, add it to the queue as long as we haven't already
        # explored it or we haven't already added it to the queue
        for rank, child in ranked_children:
            if child in closed_set or child in search_queue_set:
                continue
            else:
                search_stack.append(child)
                search_queue_set |= {child}

    print("Search Terminated")
    return None
