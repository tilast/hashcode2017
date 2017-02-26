"""
This solution uses ideas from dynamic programming (https://www.wikiwand.com/en/Dynamic_programming)
applied to knapsack problem (https://www.wikiwand.com/en/Knapsack_problem)

Each cache can be thought of as knapsack, into which we seek to place item (video) to maximize value.
What's different from traditional formulation is that values are not constant: profit from placing video x
into cache y depends on number of requests served from cache y. This means that the value depends not only on
which cache it's placed in, but also what videos are contained in other caches: as we only serve video from
closest cache, adding video to farther cache doesn't increase score.

Algorithm:
1. shuffle caches (to try to find better order of optimization)
2. for each cache:
    2.1. Solve 0/1 knapsack problem using dynamic programming (when computing values,
    take previous allocations into account to avoid placing video into cache with higher latency,
    which doesn't improve score)
    2.2. Update solution (placement of videos into cache servers)

Runtime complexity: O(C*V*S), where C – number of cache servers, V – number of videos, S – cache size.
(this is pretty high: 'trending_today' dataset has 100 caches, 10 000 videos and 50 000 cache size;
this means that the number of operations to solve this is proportionate to 5 * 10^10)
Memory usage: O(V*S)
"""

from collections import namedtuple
from random import shuffle
import time
import sys

DC_SOURCE = -1


def read_dataset(file_name):
    with open(file_name) as f:
        lines = [l.strip() for l in f.readlines()]

        current_line = 0
        video_count, endpoint_count, request_description_count, cache_count, cache_size = \
            [int(l) for l in lines[current_line].split(' ')]

        current_line += 1
        video_sizes = [int(l) for l in lines[current_line].split()]

        current_line += 1
        endpoint_source_to_latency = dict()
        for endpoint in range(endpoint_count):
            endpoint_dc_latency, endpoint_cache_count = [int(l) for l in lines[current_line].split()]
            current_line += 1
            endpoint_source_to_latency[(endpoint, DC_SOURCE)] = endpoint_dc_latency

            for _ in range(endpoint_cache_count):
                cache, latency = [int(l) for l in lines[current_line].split()]
                current_line += 1
                endpoint_source_to_latency[(endpoint, cache)] = latency

        RequestDescription = namedtuple('RequestDescription', 'video endpoint request_count')
        request_descriptions = []
        for i in range(request_description_count):
            video, endpoint, request_count = [int(l) for l in lines[current_line].split()]
            current_line += 1
            request_descriptions.append(RequestDescription(video, endpoint, request_count))

        return (video_count, endpoint_count, request_description_count, cache_count, cache_size,
                video_sizes, endpoint_source_to_latency, request_descriptions)


def get_endpoint_to_sorted_caches(endpoint_source_to_latency):
    endpoint_to_caches = dict()

    for endpoint, source in endpoint_source_to_latency.keys():
        if source != DC_SOURCE:
            endpoint_to_caches.setdefault(endpoint, list()).append(source)

    # sort caches by ascending latency
    for endpoint, caches in endpoint_to_caches.items():
        caches.sort(key=lambda c: endpoint_source_to_latency[(endpoint, c)])

    return endpoint_to_caches


def get_video_to_endpoint_request_count_list(request_descriptions):
    EndpointRequestCount = namedtuple('EndpointRequestCount', 'endpoint request_count')
    video_to_endpoint_request_count_list = dict()

    for video, endpoint, request_count in request_descriptions:
        video_to_endpoint_request_count_list.setdefault(video, set()).add(EndpointRequestCount(endpoint, request_count))

    return video_to_endpoint_request_count_list


def knapsack01_dp(video_count, video_sizes, cache_size, cache, cache_to_videos, endpoint_source_to_latency,
                  video_to_endpoint_request_count_list, endpoint_to_sorted_caches):
    # (video_count + 1) x (cache_size + 1)
    table = [[0 for _ in range(cache_size + 1)] for _ in range(video_count + 1)]

    for v in range(1, video_count + 1):
        size = video_sizes[v - 1]
        value = compute_value(v - 1, cache, cache_to_videos, endpoint_source_to_latency,
                              video_to_endpoint_request_count_list, endpoint_to_sorted_caches)
        for s in range(1, cache_size + 1):
            if size > s:
                table[v][s] = table[v - 1][s]
            else:
                table[v][s] = max(table[v - 1][s], table[v - 1][s - size] + value)

    result_videos = set()
    s = cache_size
    for v in range(video_count, 0, -1):
        was_added = table[v][s] != table[v - 1][s]

        if was_added:
            size = video_sizes[v - 1]
            result_videos.add(v - 1)
            s -= size

    return result_videos


# faster than built-in
def max(a, b):
    if a >= b:
        return a
    else:
        return b


def compute_value(video, cache, cache_to_videos, endpoint_source_to_latency, video_to_endpoint_request_count_list,
                  endpoint_to_sorted_caches):
    value = 0
    endpoint_request_count_list = video_to_endpoint_request_count_list.get(video, list())

    for endpoint, request_count in endpoint_request_count_list:
        caches = endpoint_to_sorted_caches.get(endpoint, set())
        for c in caches:
            if video in cache_to_videos.get(c, set()):
                # video is contained in closer cache
                break
            elif c == cache:
                value += compute_profit(request_count, endpoint_source_to_latency[(endpoint, DC_SOURCE)],
                                        endpoint_source_to_latency[(endpoint, cache)])
                break

    return value


def compute_profit(request_count, dc_latency, cache_latency):
    return request_count * (dc_latency - cache_latency)


def compute_score(cache_to_videos, endpoint_source_to_latency, request_descriptions, endpoint_to_sorted_caches):
    total_profit = 0
    total_request_count = 0

    for video, endpoint, request_count in request_descriptions:
        total_request_count += request_count
        caches = endpoint_to_sorted_caches.get(endpoint, set())
        for c in caches:
            if video in cache_to_videos.get(c, set()):
                total_profit += compute_profit(request_count, endpoint_source_to_latency[(endpoint, DC_SOURCE)],
                                               endpoint_source_to_latency[(endpoint, c)])
                break

    return int(total_profit / total_request_count * 1000)


def write_result(output_file_name, cache_to_videos):
    with open(output_file_name, "w") as out:
        n = len(cache_to_videos)
        out.write(str(n))
        out.write('\n')
        for source, videos in cache_to_videos.items():
            out.write(str(source) + " " + " ".join([str(i) for i in videos]))
            out.write("\n")


def solve(input_file_name, output_file_name):
    video_count, endpoint_count, request_description_count, cache_count, cache_size, \
    video_sizes, endpoint_source_to_latency, request_descriptions = read_dataset(input_file_name)

    # precompute to speed up value computations
    endpoint_to_sorted_caches = get_endpoint_to_sorted_caches(endpoint_source_to_latency)
    video_to_endpoint_request_count_list = get_video_to_endpoint_request_count_list(request_descriptions)

    best_score = 0

    permutations_to_try = 10
    caches = [_ for _ in range(cache_count)]
    cache_permutations = []
    for _ in range(permutations_to_try):
        cache_permutation = list(caches)
        shuffle(cache_permutation)
        cache_permutations.append(cache_permutation)

    for cache_permutation in cache_permutations:
        cache_to_videos = dict()

        for cache in cache_permutation:
            print('Optimizing cache #%d' % cache)
            start_time = time.time()
            cache_videos = knapsack01_dp(video_count, video_sizes, cache_size, cache, cache_to_videos,
                                         endpoint_source_to_latency, video_to_endpoint_request_count_list,
                                         endpoint_to_sorted_caches)
            cache_to_videos[cache] = cache_videos
            print("Completed in %s seconds" % (time.time() - start_time))

        current_score = compute_score(cache_to_videos, endpoint_source_to_latency, request_descriptions,
                                      endpoint_to_sorted_caches)
        if current_score > best_score:
            best_score = current_score
            print('Score: %d' % best_score)
            write_result(output_file_name, cache_to_videos)

    print('Best score: %d' % best_score)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Dataset name should be specified as argument')
        exit(1)

    dataset_name = sys.argv[1]
    input_file_name = dataset_name + '.in'
    output_file_name = dataset_name + '.out'

    start_time = time.time()
    solve(input_file_name, output_file_name)
    print("Completed in %s seconds" % (time.time() - start_time))
