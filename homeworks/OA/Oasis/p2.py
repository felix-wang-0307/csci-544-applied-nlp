from copy import deepcopy

class Solution:

    def max_subarray(self, nums):
        if not nums:
            return None  # empty

        max_sum = current_sum = nums[0]
        start = end = temp_start = 0

        for i in range(1, len(nums)):
            # If adding nums[i] is worse than starting fresh from nums[i]
            if current_sum + nums[i] < nums[i]:
                current_sum = nums[i]
                temp_start = i
            else:
                current_sum += nums[i]

            # Update max_sum if the current sum is larger
            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i

        return start, end, max_sum

    def min_subarray(self, nums):
        nums = [-num for num in nums]
        start, end, min_sum = self.max_subarray(nums)
        return start, end, -min_sum
    
    def fuck(self, num, impactFactor):
        if num >= 0:
            return num // impactFactor
        else:
            if num % impactFactor == 0:
                return num // impactFactor
            else:
                return num // impactFactor + 1

    def calculateMaxQualityScore(self, impactFactor: int, ratings: list[int]) -> int:
        # Choice 1: amplify (*) a subsequence of ratings
        amplify_start, amplify_end, amplify_sum = self.max_subarray(ratings)
        new_ratings = deepcopy(ratings)
        for i in range(amplify_start, amplify_end + 1):
            new_ratings[i] *= impactFactor
        _, _, max_quality_score_1 = self.max_subarray(new_ratings)

        # Choice 2: adjust (/) a subsequence of ratings
        adjust_ratings = [self.fuck(rating, impactFactor) for rating in ratings]
        adjust_start, adjust_end, _ = self.min_subarray(adjust_ratings)
        new_ratings_2 = deepcopy(ratings)
        for i in range(adjust_start, adjust_end + 1):
            new_ratings_2[i] //= impactFactor
        _, _, max_quality_score_2 = self.max_subarray(new_ratings_2)

        return max(max_quality_score_1, max_quality_score_2)

if __name__ == "__main__":
    solution = Solution()
    print(solution.calculateMaxQualityScore(3, [1, 2, 3, 4, 5]))  # 15
    print(solution.calculateMaxQualityScore(2, [1, 2, 3, 4, 5]))  # 14
    print(solution.calculateMaxQualityScore(3, [5, -3, -3, 2, 4]))
    print(solution.calculateMaxQualityScore(2, [-5, -3, -3, -6, -4]))
