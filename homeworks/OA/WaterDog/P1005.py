n, m = map(int, input().split())

lines = []
for _ in range(n):
    lines.append([0] + list(map(int, input().split())))

def solve(line):
    dp = [[0] * (m + 2) for _ in range(m + 2)]  # m * m
    for i in range(1, m + 1):
        dp[i][i] = line[i]
    for span in range(2, m + 1):
        for l in range(1, m - span + 2):
            r = l + span - 1
            # remove l
            dp[l][r] = dp[l + 1][r] * 2 + line[l]
            # remove r
            dp[l][r] = max(dp[l][r], dp[l][r - 1] * 2 + line[r])

    # print(line)
    # for row in dp:
    #     print(row)
    # print()
    return dp[1][m] * 2

ans = 0
for line in lines:
    ans += solve(line)

print(ans)
    