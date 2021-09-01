## 力扣所有刷过的hard题

#### [1575. 统计所有可行路径](https://leetcode-cn.com/problems/count-all-possible-routes/)

题目：

给你一个 互不相同 的整数数组，其中 locations[i] 表示第 i 个城市的位置。同时给你 start，finish 和 fuel 分别表示出发城市、目的地城市和你初始拥有的汽油总量

每一步中，如果你在城市 i ，你可以选择任意一个城市 j ，满足  j != i 且 0 <= j < locations.length ，并移动到城市 j 。从城市 i 移动到 j 消耗的汽油量为 |locations[i] - locations[j]|，|x| 表示 x 的绝对值。

请注意， fuel 任何时刻都 不能 为负，且你 可以 经过任意城市超过一次（包括 start 和 finish ）。

请你返回从 start 到 finish 所有可能路径的数目。

由于答案可能很大， 请将它对 10^9 + 7 取余后返回。



解答：

这是一道很简单的DP，直接算肯定算不出来，我们定义dp\[i\]\[j\]表示当剩余油量为i时，以j为起始节点的路径数。

所以题解如下：

``` Java
class Solution {
    public int countRoutes(int[] locations, int start, int finish, int fuel) {
        int mod = 1000000007;
        int len = locations.length;
        // 当剩余油量为m时，到达n点的路径个数
        int[][] dp = new int[210][110];
        dp[0][start] = 1;
        int ans = 0;
        for (int k = 0; k <= fuel; ++ k) {
            for (int i = 0; i < len; ++ i) {
                for (int j = 0; j < len; ++ j) {
                    if (i == j) continue;
                    int distance = Math.abs(locations[i] - locations[j]);
                    if (distance > k) {
                        continue;
                    }
                  	// 这里是动态转移方程l
                    dp[k][i] += dp[k - distance][j];
                    dp[k][i] %= mod;
                }
            }
        }
        for (int k = 0; k <= fuel; ++ k) {
            ans += dp[k][finish];
            ans %= mod;
        }
        return ans;
    }
}
```

#### [992. K 个不同整数的子数组](https://leetcode-cn.com/problems/subarrays-with-k-different-integers/)

题目：

给定一个正整数数组 A，如果 A 的某个子数组中不同整数的个数恰好为 K，则称 A 的这个连续、不一定不同的子数组为好子数组。

（例如，[1,2,3,1,2] 中有 3 个不同的整数：1，2，以及 3。）

返回 A 中好子数组的数目。



解答：

这道题是数学中**恰好转换成最多**的问题。求恰好K个，那我求最多K个f(k)，然后用f(k) - f(k-1)不就行了吗？最多K个很容易用滑动窗口求出来。

``` Java
class Solution {
    
    private int[] rcd;

    private int size = 0;

    public int subarraysWithKDistinct(int[] nums, int k) {
        return f(nums, k) - f(nums, k - 1);
    }

    // subarraysWithKDistinct(k) = f(k) - f(k - 1);
    // f(k)表示整个数组里，最多有k个不同整数的子区间个数。
    // 恰好有k个不同整数的子区间个数=最多有k个不同整数的子区间个数-最多有k-1个不同整数的子区间个数。
    private int f(int[] nums, int k) {
        rcd = new int[nums.length + 1];
        size = 0;
        int l = 0, r = 0;
        int count = 0;
        while (r < nums.length) {
            put(nums[r++]);
            while (size > k) {
                remove(nums[l++]);
            }
            // 相当巧妙啊！这行我看了许久，首先就是对于右边界未触及的情况，所有子区间个数等于r-l的总和
            // 因为在区间[l, r)之间，所有子区间个数=(r-l) + (r-l-1) + (r-l-2) + ... + 1; 恰巧等于循环得到的(r-l)总和(因为r在增加，l不变)。
            // 而对于右边界触及的情况，此时更新l之后，在[l, r]之间，[l, r - 1]已经被计算过了，还剩[l + 1, r], [l + 2, r] ... [r - 1, r]没有计算过
            // 而这恰恰等于(r - l)，所以维护了数量。
            // 官方这么解释：所有的 左边界固定前提下，根据右边界最右的下标，计算出来的子区间的个数就是整个函数要返回的值。用右边界固定的前提下，左边界最左边的下标去计算也是完全可以的。
            count += (r - l);
        }
        return count;
    }

    private void put(int val) {
        if (rcd[val] == 0) {
            ++ size;
        }
        ++ rcd[val];
    }

    private void remove(int val) {
        -- rcd[val];
        if (rcd[val] == 0) {
            -- size;
        }
    }
}
```

#### [632. 最小区间](https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/)

题目：

你有 k 个 非递减排列 的整数列表。找到一个 最小 区间，使得 k 个列表中的每个列表至少有一个数包含在其中。

我们定义如果 b-a < d-c 或者在 b-a == d-c 时 a < c，则区间 [a,b] 比 [c,d] 小。



解答：

这道题有两种解法，一种是直接上滑动窗口，记录窗口内的元素所属数组，统计个数，达到了k即可，最后进行比较长度；第二种是优先队列，即**合并K个有序链表**，通过不断取得某一数组最小值，维持整个优先队列大小为k，即里面的元素均来自不同的数组，然后计算最大元素与最小元素差值，找到最小的即可。最大值通过max维护，最小值通过队首元素维护。

``` Java
public int[] smallestRange(List<List<Integer>> nums) {
    int n = nums.size();
    int inf = 0x3f3f3f;
    int max = -inf; // 当前最大值
    int st = -inf;  // 起点
    int ed = inf;   // 终点

    PriorityQueue<Node> pq = new PriorityQueue<>((o1, o2) -> Integer.compare(o1.val, o2.val));

    // 相当于合并k个有序链表，把 head 放进去
    for (int i = 0; i < n; i++) {
        int val = nums.get(i).get(0);
        pq.offer(new Node(i, 0, val));
        max = Math.max(max, val);
    }
    
    // 必须包含 k 个元素
    while (pq.size() == n) {
        Node node = pq.poll();
        int i = node.i;
        int j = node.j;
        int val = node.val;

        // 更新区间长度
        if (max - val < ed - st) {
            st = val;
            ed = max;
        }
        
        // 为堆中填充元素
        if (j + 1 < nums.get(i).size()) {
            int nVal = nums.get(i).get(j + 1);
            pq.offer(new Node(i, j + 1, nVal));
            max = Math.max(max, nVal);
        }
    }
    return new int[]{st, ed};

}

class Node{
    int i, j, val;

    public Node(int i, int j, int val) {
        this.i = i;
        this.j = j;
        this.val = val;
    }
}
```

#### [1312. 让字符串成为回文串的最少插入次数](https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)

题目：

给你一个字符串 `s` ，每一次操作你都可以在字符串的任意位置插入任意字符。

请你返回让 `s` 成为回文串的 **最少操作次数** 。



解答：

这道题可以使用贪心思想，即，尽可能利用已经存在的**回文子序列**，这样添加操作就只是围绕这个子序列进行对称处理，所以变成了求最长回文子序列问题，这个问题还是简单的。

``` Java
class Solution {
    public int minInsertions(String s) {
        char[] str = s.toCharArray();
        int[][] dp = new int[str.length][str.length];
        for (int i = 0; i < str.length; ++ i) {
            dp[i][i] = 1;
        }
        if (str.length == 1) {
            return 0;
        }
        int max = 0;
        for (int i = str.length - 2; i >= 0; -- i) {
            for (int j = i + 1; j < str.length; ++ j) {
                if (str[i] == str[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
                max = Math.max(max, dp[i][j]);
            }
        }
        return str.length - max;
    }
}
```

#### [154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)

题目：

已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,4,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,4]
若旋转 7 次，则可以得到 [0,1,4,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个可能存在 重复 元素值的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。



解答：

最简单粗暴的方法：直接遍历。但是肯定没这么简单，所以这题考的是二分，每次判断二分的位置是否是递增，还是递减，还是出现了转折点，即可。

``` Java
func findMin(nums []int) int {
	index := -1
	l := len(nums)
	for i := 0; i < l-1; i++ {
		if nums[i] > nums[i+1] {
			index = i
			break
		}
	}
	if index == -1 {
		if nums[l-1] < nums[0] {
			return nums[l-1]
		} else {
			return nums[0]
		}
	}
	return nums[index+1]
}
```

#### [403. 青蛙过河](https://leetcode-cn.com/problems/frog-jump/)

题目：

一只青蛙想要过河。 假定河流被等分为若干个单元格，并且在每一个单元格内都有可能放有一块石子（也有可能没有）。 青蛙可以跳上石子，但是不可以跳入水中。

给你石子的位置列表 stones（用单元格序号 升序 表示）， 请判定青蛙能否成功过河（即能否在最后一步跳至最后一块石子上）。

开始时， 青蛙默认已站在第一块石子上，并可以假定它第一步只能跳跃一个单位（即只能从单元格 1 跳至单元格 2 ）。

如果青蛙上一步跳跃了 k 个单位，那么它接下来的跳跃距离只能选择为 k - 1、k 或 k + 1 个单位。 另请注意，青蛙只能向前方（终点的方向）跳跃。



解答：

暴力模拟：记录每次可以到达的位置，然后记录这个位置能跳跃的步数，最后看能不能抵达即可。

``` Java
class Solution {
    private static class Pair {
      	// 位置下标
        int index;
      	// 指出这个位置的步数
        HashSet<Integer> hashSet = new HashSet<>();
    }

    public boolean canCross(int[] stones) {
        HashMap<Integer, Integer> indexMap = new HashMap<>();
        Pair[] pairs = new Pair[stones.length];
        for (int i = 0; i < stones.length; ++i) {
            pairs[i] = new Pair();
            pairs[i].index = stones[i];
            indexMap.put(stones[i], i);
        }
        pairs[0].hashSet.add(1);
        for (int i = 0; i < stones.length - 1; ++i) {
          	// 看一下当前位置能够到达的所有位置
            for (int step : pairs[i].hashSet) {
                if (step == 0) continue;
                int tmp = step + pairs[i].index;
                // System.out.println(i);
              	// 如果存在这样的位置，则更新这个位置能够跳的步数
                if (indexMap.containsKey(tmp)) {
                    int ii = indexMap.get(tmp);
                    pairs[ii].hashSet.add(step - 1);
                    pairs[ii].hashSet.add(step);
                    pairs[ii].hashSet.add(step + 1);
                }
            }
        }
        return !pairs[stones.length - 1].hashSet.isEmpty();
    }
}
```

#### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

题目：

给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符



解答：

典型的DP问题，其实这题乍一看我还是蛮蒙的，但是很多时候就硬DP，所以有dp\[i]\[j]为world1前i个与world2前j个转换所需的最少次数。

当他们不同时，无非就是删除：dp\[i]\[j-1] + 1；修改：dp\[i-1]\[j-1]+1；添加：dp\[i-1]\[j]+1之中选择。

``` Java
func min(x, y int) int {
	if x < y {
		return x
	} else {
		return y
	}
}

func minDistance(word1 string, word2 string) int {
	str1 := []byte(word1)
	str2 := []byte(word2)
	len1 := len(str1)
	len2 := len(str2)
	var dp [510][510]int
	if len1 == 0 {
		return len2
	} else if len2 == 0 {
		return len1
	} else {
		for i := 1; i <= len1; i ++ {
			dp[i][0] = i
		}
		for i := 1; i <= len2; i ++ {
			dp[0][i] = i
		}
	}
	for i := 1; i <= len1; i ++ {
		for j := 1; j <= len2; j++ {
			if str1[i-1] == str2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
                // 在不同时，选择编辑/添加/删除选一个
				dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1
			}
		}
	}
	return dp[len1][len2]
}
```

#### [1269. 停在原地的方案数](https://leetcode-cn.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/)

题目：

有一个长度为 arrLen 的数组，开始有一个指针在索引 0 处。

每一步操作中，你可以将指针向左或向右移动 1 步，或者停在原地（指针不能被移动到数组范围外）。

给你两个整数 steps 和 arrLen ，请你计算并返回：在恰好执行 steps 次操作以后，指针仍然指向索引 0 处的方案数。

由于答案可能会很大，请返回方案数 模 10^9 + 7 后的结果。



解答：DP大法好啊！直接定义dp\[i]\[j]表示移动了i次，在j位置的方案数。则dp\[i]\[j] = dp\[i-1]\[j] + dp\[i-1]\[j-1] + dp\[i-1]\[j+1]；此时注意边界检查。

``` Java
class Solution {
    public int numWays(int steps, int arrLen) {
        final int MODULO = 1000000007;
        int maxColumn = Math.min(arrLen - 1, steps);
        int[][] dp = new int[steps + 1][maxColumn + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= steps; i++) {
            for (int j = 0; j <= maxColumn; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j - 1 >= 0) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MODULO;
                }
                if (j + 1 <= maxColumn) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][j + 1]) % MODULO;
                }
            }
        }
        return dp[steps][0];
    }
}
```

#### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

题目：

给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png)



解答：

不算很典型的单调栈题目，因为想要接住雨水，必须找到后面一个大于等于它的柱子才行，而找到第一个大于等于/小于等于/大于/小于场景正是单调栈可以做的事，唯一麻烦的就是处理起来细节比较繁琐，因为我们不能单纯地找到柱子，还要计算间距和高度差。

``` Java
class Solution {
    private static class Pair {
        int index;
        int value;
    }

    private final LinkedList<Pair> list = new LinkedList<>();

    private int ans = 0;

    private void put(Pair pair) {
        if (!list.isEmpty()) {
            Pair first = list.getFirst();
            while (pair.value > first.value) {
                Pair tmp = list.getFirst();
                list.removeFirst();
                first = list.peekFirst();
                if (first == null) {
                    break;
                } else {
                    ans += (Math.min(pair.value, first.value) - tmp.value) * (pair.index - first.index - 1);
                }
            }
        }
        list.addFirst(pair);
    }

    public int trap(int[] height) {
        Pair[] pairs = new Pair[height.length];
        int begin = -1;
        for (int i = 0; i < height.length; ++i) {
            pairs[i] = new Pair();
            pairs[i].index = i;
            pairs[i].value = height[i];
            if (height[i] != 0 && begin == -1) {
                begin = i;
            }
        }
        if (begin == -1) {
            return 0;
        }
        for (int i = begin; i < height.length; ++i) {
            put(pairs[i]);
        }
        return ans;
    }
}
```

#### [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

题目：

给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

k 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

进阶：

你可以设计一个只使用常数额外空间的算法来解决此问题吗？
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex1.jpg)



解答：

模拟题，就硬模拟。

``` Java
class Solution {

    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode root = new ListNode();
        ListNode tmp = root;
        int i = k;
        while (head != null) {
            ListNode from = head, to = null;
            while (head != null && i > 0) {
                to = head;
                head = head.next;
                -- i;
            }
            if (head == null && i > 0) {
                root.next = from;
            } else {
                to.next = null;
                ff(from);
                root.next = to;
                root = from;
            }
            i = k;
        }
        return tmp.next;
    }

    private ListNode ff(ListNode node) {
        if (node.next == null) {
            return node;
        }
        ListNode prev = ff(node.next);
        node.next = null;
        prev.next = node;
        return node;
    }
}
```

#### [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

题目：

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。



解答：

这题考的是活用归并排序，基于这样一个思想：划分出mid位置，mid之前全部是有序的，mid后面也是有序的；然后判断，nums\[a]与nums\[b]之间的关系，其中a <= mid < b。如果nums\[a] > nums\[b]，则mid - a这个范围内的数字都可以和nums\[b]凑成逆序对。不过记得在处理过程中同时合并两个有序数组。

``` Java
class Solution {
    public int reversePairs(int[] nums) {
        mergeSort(nums, 0, nums.length - 1);
        return ans;
    }

    private int ans = 0;

    private void mergeSort(int[] nums, int from, int to) {
        if (from >= to) {
            return;
        }
        int mid = (from + to) / 2;
        mergeSort(nums, from, mid);
        mergeSort(nums, mid + 1, to);
        int a = from, b = mid + 1;
        int k = 0;
        int[] tmp = new int[to - from + 1];
        while (a <= mid) {
            while (b <= to) {
                if (nums[b] < nums[a]) {
                    // System.out.println(mid + " " + a + " " + b);
                    // System.out.println(nums[a] + " | " + nums[b]);
                    tmp[k++] = nums[b++];
                    ans += (mid - a + 1);
                } else {
                    break;
                }
            }
            tmp[k++] = nums[a++];
        }
        while (b <= to) {
            // System.out.println(from + " # " + to);
            // System.out.println(k + " : " + tmp.length);
            tmp[k++] = nums[b++];
        }
        if (to + 1 - from >= 0) System.arraycopy(tmp, 0, nums, from, to + 1 - from);
    }
}
```

#### [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

题目：

给你一个字符串表达式 `s` ，请你实现一个基本计算器来计算并返回它的值。



解答：

模拟题，和计算逆波兰表达式有点类似，通过栈解决。

``` Java
class Solution {
    private final LinkedList<Integer> numbers = new LinkedList<>();

    private final LinkedList<Character> operators = new LinkedList<>();

    private Item[] items;

    private static class Item {
        private int num;

        private char op;

        private boolean type;
    }

    public int calculate(String s) {
        char[] tmpStr = s.toCharArray();
        int len = 0;
        for (int i = 0; i < tmpStr.length; ++ i) {
            if (tmpStr[i] != ' ') {
                tmpStr[len++] = tmpStr[i];
            }
        }
        String input = new String(tmpStr, 0, len);
        if (input.charAt(0) == '-') {
            input = "0" + input;
        }
        String s1 = "\\(-", s2 = "(0-";
        String s3 = "\\(+", s4 = "(0";
        input = input.replaceAll(s1, s2).replaceAll(s3, s4);
        char[] str = input.toCharArray();
        convert(str, input);
        for (int i = 0; i < items.length;) {
            if (!items[i].type) {
                putNum(items[i++].num);
            } else {
                putOps(items[i++].op);
                if (i < items.length && !items[i].type) {
                    putNum(items[i++].num);
                }
            }
            // log();
        }
        compute();
        return numbers.getFirst();
    }

    @SuppressWarnings("unchecked")
    private void log() {
        int len = Math.max(numbers.size(), operators.size());
        var a = (LinkedList<Integer>) numbers.clone();
        var b = (LinkedList<Character>) operators.clone();
        for (int i = 0; i < len; ++ i) {
            if (i >= numbers.size()) {
                System.out.println("      " + b.getFirst());
                b.removeFirst();
            } else if (i >= operators.size()) {
                System.out.println(a.getFirst() + "      ");
                a.removeFirst();
            } else {
                System.out.println(a.getFirst() + "     " + b.getFirst());
                a.removeFirst();
                b.removeFirst();
            }
        }
        System.out.println("#######");
    }

    private void convert(char[] str, String s) {
        Item[] tmpItems = new Item[str.length];
        int len = 0;
        for (int i = 0; i < str.length;) {
            Item item = new Item();
            if (str[i] >= '0' && str[i] <= '9') {
                int k = i;
                while (i < str.length && str[i] >= '0' && str[i] <= '9') ++ i;
                item.num = Integer.parseInt(s.substring(k, i));
                item.type = false;
                tmpItems[len++] = item;
            } else {
                item.op = str[i++];
                item.type = true;
                tmpItems[len++] = item;
            }
        }
        items = new Item[len];
        System.arraycopy(tmpItems, 0, items, 0, len);
    }

    private void compute() {
        char top;
        int v2, v1;
        boolean flag = false;
        while (!operators.isEmpty()) {
            top = operators.getFirst();
            if (top == '(') {
                if (flag) {
                    operators.removeFirst();
                }
            } else {
                operators.removeFirst();
            }
            if (top == ')') {
                flag = true;
                continue;
            }
            if (top == '(') {
                break;
            }
            v2 = numbers.getFirst();
            numbers.removeFirst();
            v1 = numbers.getFirst();
            numbers.removeFirst();
            // System.out.println("c:" + v1 + " " + top + " " + v2);
            int v;
            switch (top) {
                case '+' -> v = v1 + v2;
                case '-' -> v = v1 - v2;
                case '*' -> v = v1 * v2;
                case '/' -> v = v1 / v2;
                default -> v = 0;
            }
            numbers.addFirst(v);
        }
    }

    private void putNum(int num) {
        numbers.addFirst(num);
    }

    private void putOps(char op) {
        if (operators.isEmpty()) {
            operators.addFirst(op);
            // log();
        } else {
            if (op == '(') {
                operators.addFirst(op);
                // log();
            } else if (op == ')') {
                operators.addFirst(op);
                // log();
                compute();
            } else {
                if (lower(op, operators.getFirst())) {
                    operators.addFirst(op);
                    // log();
                } else {
                    // log();
                    compute();
                    operators.addFirst(op);
                }
            }
        }
    }

    private boolean lower(char a, char b) {
        if (a == '*' || a == '/') {
            return b == '+' || b == '-';
        } else {
            return false;
        }
    }
}
```

#### [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

题目：

路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。



解答：

就硬遍历，硬模拟。

``` Java
class Solution {
    private int ans = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        find(root);
        return ans;
    }

    private int find(TreeNode parent) {
        if (parent == null) {
            return Integer.MIN_VALUE;
        }
        ans = Math.max(ans, parent.val);
        int left = find(parent.left);
        int right = find(parent.right);
        if (parent.val >= 0) {
            if (left >= 0) {
                if (right >= 0) {
                    ans = Math.max(ans, parent.val + left + right);
                    return parent.val + Math.max(left, right);
                } else {
                    ans = Math.max(ans, parent.val + left);
                    return parent.val + left;
                }
            } else {
                if (right >= 0) {
                    ans = Math.max(ans, parent.val + right);
                    return parent.val + right;
                } else {
                    ans = Math.max(ans, parent.val);
                    return parent.val;
                }
            }
        } else {
            if (left >= 0) {
                if (right >= 0) {
                    ans = Math.max(ans, left);
                    ans = Math.max(ans, right);
                    ans = Math.max(ans, parent.val + left + right);
                    int v1 = parent.val + left;
                    int v2 = parent.val + right;
                    int max = Math.max(v1, v2);
                    return max;
                } else {
                    ans = Math.max(ans, left);
                    int v = parent.val + left;
                    return v;
                }
            } else {
                if (right >= 0) {
                    ans = Math.max(ans, right);
                    int v = parent.val + right;
                    return v;
                } else {
                    return parent.val;
                }
            }
        }
    }
}
```

#### [32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

题目：

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。



解答：

这道题还是很有意思的，其实一开始大家都想得到使用栈，这没错，但是我看评论区才发现我们可以通过栈，把所有合法的括号`(` `)`标记成1，然后统计连续的1的长度即可。

``` Java
class Solution {
    private static class Pair {
        char val;
        int index;
    }

    public int longestValidParentheses(String s) {
        char[] str = s.toCharArray();
        int[] dp = new int[str.length];
        Pair[] pairs = new Pair[str.length];
        LinkedList<Pair> stack = new LinkedList<>();
        for (int i = 0; i < str.length; ++ i) {
            pairs[i] = new Pair();
            pairs[i].val = str[i];
            pairs[i].index = i;
        }
        for (Pair pair : pairs) {
            if (pair.val == '(') {
                stack.addFirst(pair);
            } else {
                if (!stack.isEmpty()) {
                    Pair tmp = stack.getFirst();
                    stack.removeFirst();
                    dp[tmp.index] = 1;
                    dp[pair.index] = 1;
                }
            }
        }
        int ans = 0, count = 0;
        for (int a : dp) {
            if (a == 1) {
                ++ count;
            } else {
                count = 0;
            }
            ans = Math.max(ans, count);
        }
        return ans;
    }
}
```

#### [664. 奇怪的打印机](https://leetcode-cn.com/problems/strange-printer/)

题目：

有台奇怪的打印机有以下两个特殊要求：

* 打印机每次只能打印由 同一个字符 组成的序列。
* 每次可以在任意起始和结束位置打印新字符，并且会覆盖掉原来已有的字符。

给你一个字符串 s ，你的任务是计算这个打印机打印它需要的最少打印次数。



解答：

这道题其实如果可以想到DP还是很简单的，直接二维DP，i, j表示从i到j位置的字符串的打印次数。其实到这里我们可以总结一下，一般涉及到字符串DP的题目，都是内循环，且如果涉及到最值问题，一般都是DP，区间最值则是拆分成循环比较。



内循环：

``` Java
for (int i = 0; i < len; ++ i) {
    for (int j = i; j >= 0; -- j) {
      	dp[j][i] = ...
    }
}
```

切分求值：

``` Java
for (int i = 0; i < len; ++ i) {
    for (int j = i; j >= 0; -- j) {
      	for (int k = i-1; k >= j; -- k) {
          dp[j][i] = max/min(dp[j][k], dp[k+1][i]);
        }
    }
}
```



所以我们有题解：

``` Java
class Solution {

    public int strangePrinter(String s) {
        char[] str = s.toCharArray();
        int[][] dp = new int[str.length + 1][str.length + 1];
        for (int i = 0; i < str.length; ++ i) {
            dp[i][i] = 1;
        }
        dp[0][0] = 1;
        for (int i = 0; i < str.length; ++ i) {
            for (int j = i-1; j >= 0; -- j) {
                if (str[i] == str[j]) {
                    dp[j][i] = dp[j][i-1];
                } else {
                    dp[j][i] = Integer.MAX_VALUE;
                    for (int k = i-1; k >= j; -- k) {
                        // 字符串一般使用切分DP求最值
                        dp[j][i] = Math.min(dp[j][i], dp[j][k] + dp[k+1][i]);
                    }
                }
            }
        }
        return dp[0][str.length-1];
    }
}
```

#### [140. 单词拆分 II](https://leetcode-cn.com/problems/word-break-ii/)

题目：

给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。

说明：

分隔时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。



解答：

这道题其实很简单，就是模拟题，但是为了优化我们提前记录一下，哪些区间的单词是存在字典中的，然后回溯处理即可。

``` Java
class Solution {

    private final HashSet<String> hashSet = new HashSet<>();

    private String str;

    private boolean[][] dp;

    public List<String> wordBreak(String s, List<String> wordDict) {
        str = s;
        dp = new boolean[s.length()][s.length()];
        hashSet.addAll(wordDict);
        for (int i = 0; i < s.length(); ++ i) {
            for (int j = i; j < s.length(); ++ j) {
                String tmpStr = s.substring(i, j + 1);
                if (hashSet.contains(tmpStr)) {
                    // System.out.println(i + " : " + j);
                    dp[i][j] = true;
                } else {
                    dp[i][j] = false;
                }
            }
        }
        List<List<String>> ans = f(0, s.length() - 1);
        if (ans == null) {
            return new LinkedList<>();
        }
        List<String> ans0 = new LinkedList<>();
        for (List<String> a : ans) {
            StringBuilder stringBuilder = new StringBuilder();
            for (String b : a) {
                stringBuilder.append(b).append(" ");
            }
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
            ans0.add(stringBuilder.toString());
        }
        return ans0;
    }

    public List<List<String>> f(int from, int to) {
        if (from > to) {
            return null;
        }
        List<List<String>> ans = new LinkedList<>();
        for (int i = from; i <= to; ++ i) {
            if (i == to) {
                if (dp[from][to]) {
                    // 特殊处理一下，因为此时就不需要递归了。
                    List<String> tmp = new LinkedList<>();
                    tmp.add(str.substring(from, to + 1));
                    ans.add(tmp);
                }
            } else {
                if (dp[from][i]) {
                    List<List<String>> tmpList = f(i + 1, to);
                    if (tmpList != null) {
                        for (List<String> a : tmpList) {
                            List<String> tmp = new LinkedList<>();
                            tmp.add(str.substring(from, i + 1));
                            tmp.addAll(a);
                            ans.add(tmp);
                        }
                    }
                }
            }
        }
        if (ans.isEmpty()) {
            return null;
        } else {
            return ans;
        }
    }
}
```

#### [879. 盈利计划](https://leetcode-cn.com/problems/profitable-schemes/)

题目：

集团里有 n 名员工，他们可以完成各种各样的工作创造利润。

第 i 种工作会产生 profit[i] 的利润，它要求 group[i] 名成员共同参与。如果成员参与了其中一项工作，就不能参与另一项工作。

工作的任何至少产生 minProfit 利润的子集称为 盈利计划 。并且工作的成员总数最多为 n 。

有多少种计划可以选择？因为答案很大，所以 返回结果模 10^9 + 7 的值。



解答：

最值问题，和选取问题，一般都是背包问题，这题是很典型的三维背包问题，我在这里放一个背包通项：

``` Java
public class Bag {

    // 01背包
    private int package01(int w, int[] ws, int[] vs) {
      	// 从前i个中选，当重量不超过j时的最大价值
        int[][] dp1 = new int[ws.length][w + 1];
        for (int i = w; i >= ws[0]; -- i) {
            dp1[0][i] = vs[0];
        }
        for (int i = 1; i < ws.length; ++ i) {
            // 个人感觉，这里正序倒序应该都可以
            for (int j = w; j >= 0; --j) {
                // if ...
                dp1[i][j] = Math.max(dp1[i-1][j], dp1[i-1][j-ws[i]] + vs[i]);
            }
        }
        // 化为一维
        int[] dp2 = new int[w + 1];
        for (int i = 0; i < ws.length; ++ i) {
            for (int j = w; j >= 0; -- j) {
                // if ...
                dp2[j] = Math.max(dp2[j], dp2[j-ws[i]] + vs[i]);
            }
        }
        return dp2[w];
    }

    // 完全背包
    private int packageMulti(int w, int[] ws, int[] vs) {
        int[][] dp1 = new int[ws.length][w + 1];
        for (int j = ws[0]; j <= w; ++ j) {
            dp1[0][j] = Math.max(dp1[0][j], dp1[0][j-ws[0]] + vs[0]);
        }
        for (int i = 1; i < ws.length; ++ i) {
            // 为什么这里是正序，其实很好理解，就是我的dp[i][j]依赖于dp[i][k]的结果，这里的k < j，所以我们只能是正序
            for (int j = 0; j <= w; ++ j) {
                dp1[i][j] = Math.max(dp1[i-1][j], dp1[i][j-ws[i]] + vs[i]);
            }
        }
        int[] dp2 = new int[w + 1];
        for (int i = 0; i < ws.length; ++ i) {
            for (int j = 0; j <= w; ++ j) {
                dp2[j] = Math.max(dp2[j], dp2[j-ws[i]] + vs[i]);
            }
        }
        return dp2[w];
    }
}
```

一般来说，01背包强调要么不选，要么只选一个；完全背包强调要么不选，要么可以选无数个。此外如果每个物品个数都有限制，那只能再来一个循环进行判断，没有什么好的解法。



此外，背包问题的循环处理，一般会把限制层放在内层，也就是把容量，重量，成本等限制选择的循环放在第二个循环。



现在我们来看看这题解答：

``` Java
class Solution {
    public int profitableSchemes(int n, int minProfit, int[] group, int[] profit) {
        int mod = (int) 1e9+7;
        // 利润不大于minProfit的方案数
        int[][][] dp = new int[110][110][110];
        for (int i = 0; i <= group.length; ++ i) {
            for (int j = 0; j <= n; ++ j) {
                dp[i][j][0] = 1;
            }
        }
        for (int i = 1; i <= group.length; ++ i) {
            for (int j = 0; j <= n; ++ j) {
                for (int k = 0; k <= minProfit; ++ k) {
                    if (j >= group[i-1]) {
                        if (k >= profit[i-1]) {
                            dp[i][j][k] = (dp[i-1][j][k] + dp[i-1][j-group[i-1]][k-profit[i-1]]) % mod;
                        } else {
                            dp[i][j][k] = (dp[i-1][j][k] + dp[i-1][j-group[i-1]][0]) % mod;
                        }
                    } else {
                        dp[i][j][k] = dp[i-1][j][k];
                    }
                }
            }
        }
        return dp[group.length][n][minProfit];
    }
}
```

#### [1449. 数位成本和为目标值的最大数字](https://leetcode-cn.com/problems/form-largest-integer-with-digits-that-add-up-to-target/)

题目：

给你一个整数数组 cost 和一个整数 target 。请你返回满足如下规则可以得到的 最大 整数：

给当前结果添加一个数位（i + 1）的成本为 cost[i] （cost 数组下标从 0 开始）。
总成本必须恰好等于 target 。
添加的数位中没有数字 0 。
由于答案可能会很大，请你以字符串形式返回。

如果按照上述要求无法得到任何整数，请你返回 "0" 。



解答：

这道题很容易看出来是完全背包问题。现在我们来看看该题：

``` Java
class Solution {
    private int compare(String a, String b) {
        if (a.charAt(0) == '-') {
            return -1;
        } else if (b.charAt(0) == '-') {
            return 1;
        }
        if (a.length() > b.length()) {
            return 1;
        } else if (a.length() < b.length()) {
            return -1;
        } else {
            return a.compareTo(b);
        }
    }

    public String largestNumber(int[] cost, int target) {
        String[] dp = new String[target+1];
        Arrays.fill(dp, "-1");
        dp[0] = "";
        for (int i = 1; i <= cost.length; ++ i) {
            for (int j = 0; j <= target; ++ j) {
                if (j >= cost[i-1]) {
                    String prev = dp[j];
                    String tmpStr = dp[j-cost[i-1]];
                    String tmpS = i + "";
                    if (tmpStr.length() == 0) {
                        dp[j] = tmpS;
                    } else {
                        String s1 = tmpS + tmpStr;
                        String s2 = tmpStr + tmpS;
                        if (s1.charAt(0) != '-' && s2.charAt(0) != '-') {
                            if (compare(s1, s2) > 0) {
                                if (compare(s1, prev) > 0) {
                                    dp[j] = s1;
                                }
                            } else {
                                if (compare(s2, prev) > 0) {
                                    dp[j] = s2;
                                }
                            }
                        }
                    }
                }
                // System.out.print(dp[j] + " ");
            }
            // System.out.println();
        }
        return dp[target].charAt(0) == '-' ? "0" : dp[target];
    }
}
```

#### [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

题目：

字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列：

序列中第一个单词是 beginWord 。
序列中最后一个单词是 endWord 。
每次转换只能改变一个字母。
转换过程中的中间单词必须是字典 wordList 中的单词。
给你两个单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0。



解答：

这道题是很典型的BFS搜索问题，因为每次只能改变一个字母，而一共有26个字母，我们就可以对单词的每个字母进行从a-z的替换，然后寻找最短路径。



其实说到最短路径，第一时间肯定还是想到了Dijkstra算法，但是当路径权值相同时，Dijkstra算法就可以退化成BFS，而BFS肯定写起来快，所以我们这题就是这么干的。此外，既然提到了Dijkstra，我们就来理一下最短路算法：



* 设置起始位置为第一个节点。
* 遍历所有未标记节点，找到离起始节点最近的节点k。
* 以k为中心，通过k更新它所能到达的节点与起始节点的距离
* 把k标记为以处理。
* 重复第二步。



然后此时我们来看看这题，使用BFS处理：

``` Java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        HashMap<String, Integer> dp = new HashMap<>(wordList.size() + 2);
        HashMap<String, Boolean> rcd = new HashMap<>(wordList.size() + 2);
        for (String s : wordList) {
            dp.put(s, Integer.MAX_VALUE);
            rcd.put(s, false);
        }
        if (!dp.containsKey(endWord)) {
            return 0;
        }
        dp.put(endWord, Integer.MAX_VALUE);
        rcd.put(endWord, false);
        dp.put(beginWord, 1);
        rcd.put(beginWord, false);
        LinkedList<String> queue = new LinkedList<>();
        queue.addFirst(beginWord);
        while (!queue.isEmpty()) {
            String first = queue.getFirst();
            rcd.put(first, true);
            queue.removeFirst();
            char[] str = first.toCharArray();
            for (int i = 0; i < str.length; ++ i) {
                char origin = str[i];
                for (int j = 0; j < 26; ++ j) {
                    if ('a' + j == origin) {
                        continue;
                    }
                    str[i] = (char) ('a' + j);
                    String tmpStr = new String(str);
                    if (dp.containsKey(tmpStr)) {
                        dp.put(tmpStr, Math.min(dp.get(first) + 1, dp.get(tmpStr)));
                        if (!rcd.get(tmpStr)) {
                            rcd.put(tmpStr, true);
                            queue.addLast(tmpStr);
                        }
                        // System.out.println(tmpStr + "," + dp.get(tmpStr));
                    }
                }
                str[i] = origin;
            }
        }
        return dp.get(endWord) == Integer.MAX_VALUE ? 0 : dp.get(endWord);
    }
}
```

#### [126. 单词接龙 II](https://leetcode-cn.com/problems/word-ladder-ii/)

题目：

按字典 wordList 完成从单词 beginWord 到单词 endWord 转化，一个表示此过程的 转换序列 是形式上像 beginWord -> s1 -> s2 -> ... -> sk 这样的单词序列，并满足：

每对相邻的单词之间仅有单个字母不同。
转换过程中的每个单词 si（1 <= i <= k）必须是字典 wordList 中的单词。注意，beginWord 不必是字典 wordList 中的单词。
sk == endWord
给你两个单词 beginWord 和 endWord ，以及一个字典 wordList 。请你找出并返回所有从 beginWord 到 endWord 的 最短转换序列 ，如果不存在这样的转换序列，返回一个空列表。每个序列都应该以单词列表 [beginWord, s1, s2, ..., sk] 的形式返回。



解答：

这题和上一题最大的不同是，要求记录路径。一般对于最点路径走法的记录，我们使用倒叙记录，即记录当前节点是从哪个节点过来的；然后复原的话，反过来找到路径即可。

``` Java
class Solution {
    List<List<String>> ans = new LinkedList<>();

    String bgnWord;

    HashMap<String, LinkedList<String>> paths;

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        HashMap<String, Integer> dp = new HashMap<>(wordList.size() + 2);
        HashMap<String, Boolean> rcd = new HashMap<>(wordList.size() + 2);
        paths = new HashMap<>(wordList.size() + 2);
        bgnWord = beginWord;
        for (String s : wordList) {
            dp.put(s, Integer.MAX_VALUE);
            rcd.put(s, false);
            paths.put(s, new LinkedList<>());
        }
        if (!dp.containsKey(endWord)) {
            return new LinkedList<>();
        }
        dp.put(endWord, Integer.MAX_VALUE);
        rcd.put(endWord, false);
        paths.put(endWord, new LinkedList<>());
        dp.put(beginWord, 1);
        rcd.put(beginWord, false);
        paths.put(beginWord, new LinkedList<>());
        LinkedList<String> queue = new LinkedList<>();
        queue.addFirst(beginWord);
        while (!queue.isEmpty()) {
            String first = queue.getFirst();
            rcd.put(first, true);
            queue.removeFirst();
            char[] str = first.toCharArray();
            for (int i = 0; i < str.length; ++ i) {
                char origin = str[i];
                for (int j = 0; j < 26; ++ j) {
                    if ('a' + j == origin) {
                        continue;
                    }
                    str[i] = (char) ('a' + j);
                    String tmpStr = new String(str);
                    if (dp.containsKey(tmpStr)) {
                        int a = dp.get(first) + 1;
                        int b = dp.get(tmpStr);
                        dp.put(tmpStr, Math.min(a, b));
                        if (!rcd.get(tmpStr)) {
                            rcd.put(tmpStr, true);
                            queue.addLast(tmpStr);
                        }
                        if (a < b) {
                            paths.get(tmpStr).clear();
                            paths.get(tmpStr).addLast(first);
                        } else if (a == b) {
                            paths.get(tmpStr).addLast(first);
                        }
                    }
                }
                str[i] = origin;
            }
        }
        LinkedList<String> tmp = new LinkedList<>();
        f(tmp, endWord);
        return ans;
    }

    private void f(LinkedList<String> list, String str) {
        // System.out.println(str + "," + paths.get(str));
        if (str.equals(bgnWord)) {
            list.addFirst(str);
            ans.add((List<String>) list.clone());
            list.removeFirst();
        }
        list.addFirst(str);
        for (String s : paths.get(str)) {
            f(list, s);
        }
        list.removeFirst();
    }
}
```

#### [871. 最低加油次数](https://leetcode-cn.com/problems/minimum-number-of-refueling-stops/)

题目：

汽车从起点出发驶向目的地，该目的地位于出发位置东面 target 英里处。

沿途有加油站，每个 station[i] 代表一个加油站，它位于出发位置东面 station[i][0] 英里处，并且有 station[i][1] 升汽油。

假设汽车油箱的容量是无限的，其中最初有 startFuel 升燃料。它每行驶 1 英里就会用掉 1 升汽油。

当汽车到达加油站时，它可能停下来加油，将所有汽油从加油站转移到汽车中。

为了到达目的地，汽车所必要的最低加油次数是多少？如果无法到达目的地，则返回 -1 。

注意：如果汽车到达加油站时剩余燃料为 0，它仍然可以在那里加油。如果汽车到达目的地时剩余燃料为 0，仍然认为它已经到达目的地。



解答：

这道题其实蛮艹的，因为它解法唯一，只能用优先队列求解，而且思路也是唯一的，作为一个引入优先队列的题目是不错的，但是作为一个通用算法题，可能不是很有意思。



我们通过优先队列，记录汽车行驶过的所有的加油站，如果汽油不够了，我们从经过的所有加油站取一个能加最多油的即可；然后继续。

``` Java
class Solution {
    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>((o1, o2) -> -1*Integer.compare(o1, o2));
        int[][] rcd = new int[stations.length + 2][2];
        rcd[0][0] = 0;
        rcd[0][1] = startFuel;
        rcd[rcd.length - 1][0] = target;
        rcd[rcd.length - 1][1] = 0;
        for (int i = 1; i < rcd.length - 1; ++i) {
            rcd[i][0] = stations[i - 1][0];
            rcd[i][1] = stations[i - 1][1];
        }
        int prev = rcd[0][0];
        int curr = startFuel;
        int count = 0;
        for (int i = 1; i < rcd.length; ++ i) {
            int dis = rcd[i][0] - prev;
            curr -= dis;
            // System.out.println(i);
            while (curr < 0) {
                if (priorityQueue.isEmpty()) {
                    return -1;
                }
                curr += priorityQueue.poll();
                // System.out.println("curr: " + curr);
                ++ count;
            }
            priorityQueue.add(rcd[i][1]);
            prev = rcd[i][0];
        }
        return count;
    }
}
```

#### [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

题目：

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。



解答：

优先队列解决，因为涉及到多路排序问题，此时优先队列是一个不错的方案。

``` Java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>();
        for (ListNode head : lists) {
            ListNode tmp = head;
            while (tmp != null) {
                priorityQueue.add(tmp.val);
                tmp = tmp.next;
            }
        }
        ListNode listNode = new ListNode();
        ListNode head = listNode;
        ListNode prev = null;
        while (!priorityQueue.isEmpty()) {
            listNode.val = priorityQueue.poll();
            listNode.next = new ListNode();
            prev = listNode;
            listNode = listNode.next;
        }
        if (prev == null) {
            return null;
        } else {
            prev.next = null;
            return head;
        }
    }
}
```

#### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

题目：

给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。



解答：

这道题涉及到了范围更新，或者动态更新找最值，其实结合[数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)这一题，我们基本可以知道，涉及到数据动态更新，求最值的，基本都是滑动窗口。



此题利用了延迟删除的特性，即，获取最大值之前先判断最大值是不是已经不在滑动窗口中了，如果不在我们需要剔除，并获得下一个最大值。关于**延迟删除**，其实还有另一个更加经典的题[滑动窗口的中位数](https://leetcode-cn.com/problems/sliding-window-median/)这一题。

``` Java
class Solution {

    private static class Pair {
        int val;
        int index;

        public Pair() {
        }

        public Pair(int val, int index) {
            this.val = val;
            this.index = index;
        }
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        PriorityQueue<Pair> priorityQueue = new PriorityQueue<>((v1, v2) -> {
            if (v1.val > v2.val) {
                return -1;
            } else if (v1.val < v2.val) {
                return 1;
            } else {
                // 数值一样时，选择下标最大的，这样可以尽可能用到可用的最大值
                return v1.index > v2.index ? -1 : 0;
            }
        });
        int[] tmp = new int[nums.length];
        int index = 0;
        for (int i = 0; i < k; ++ i) {
            priorityQueue.add(new Pair(nums[i], i));
        }
        tmp[index++] = priorityQueue.peek().val;
        for (int i = k; i < nums.length; ++ i) {
            priorityQueue.add(new Pair(nums[i], i));
            // 去除所有边界之外的最大值
            while (priorityQueue.peek().index <= i-k) {
                priorityQueue.poll();
            }
            tmp[index++] = priorityQueue.peek().val;
        }
        int[] ans = new int[index];
        System.arraycopy(tmp, 0, ans, 0, index);
        return ans;
    }
}
```

#### [407. 接雨水 II](https://leetcode-cn.com/problems/trapping-rain-water-ii/)

题目：

给你一个 `m x n` 的矩阵，其中的值均为非负整数，代表二维高度图每个单元的高度，请计算图中形状最多能接多少体积的雨水。

 

**示例 1:**

![img](https://assets.leetcode.com/uploads/2021/04/08/trap1-3d.jpg)



解答：

二维接雨水，我们用的是单调栈，三维我们想要接住雨水，恐怕需要上优先队列了，啥意思呢？就是说我们此时，想要接住雨水，需要找到最矮的柱子，而这个最矮的柱子就需要优先队列进行查找了。

``` Java
class Solution {

    private int m, n;

    private int[][] heightMap;

    private boolean[][] marked;

    private final int[] X = {1, 0, -1, 0};

    private final int[] Y = {0, 1, 0, -1};

    private int ans = 0;

    public int trapRainWater(int[][] heightMap0) {
        heightMap = heightMap0;
        m = heightMap0.length;
        n = heightMap0[0].length;
        marked = new boolean[m][n];
        if (m < 3 || n < 3) {
            return 0;
        }
        PriorityQueue<Pair> priorityQueue = new PriorityQueue<>(Comparator.comparingInt(p -> p.val));
        int k = Math.max(m, n);
        for (int a = 0; a < k; ++a) {
            // 边缘入
            for (int i = a; i < m - a; ++i) {
                if (i < m && a < n && !marked[i][a]) {
                    priorityQueue.add(new Pair(heightMap[i][a], i, a));
                    marked[i][a] = true;
                }
                if (i < m && n-a-1 < n && n-a-1 >= 0 && !marked[i][n-a-1]) {
                    priorityQueue.add(new Pair(heightMap[i][n - a - 1], i, n - a - 1));
                    marked[i][n - a - 1] = true;
                }
            }
            for (int i = a; i < n - a; ++i) {
                if (a < m && i < n && !marked[a][i]) {
                    priorityQueue.add(new Pair(heightMap[a][i], a, i));
                    marked[a][i] = true;
                }
                if (m-a-1 < m && m-a-1 >= 0 && i < n && !marked[m-a-1][i]) {
                    priorityQueue.add(new Pair(heightMap[m - a - 1][i], m - a - 1, i));
                    marked[m - a - 1][i] = true;
                }
            }
            while (!priorityQueue.isEmpty()) {
                Pair p = priorityQueue.poll();
                // System.out.println(p);
                for (int i = 0; i < 4; ++i) {
                    int x = p.x + X[i];
                    int y = p.y + Y[i];
                    if (check(x, y)) {
                        marked[x][y] = true;
                        if (p.val > heightMap[x][y]) {
                            ans += (p.val - heightMap[x][y]);
                            // System.out.println(x + "," + y + "=" + (p.val - heightMap[x][y]));
                            heightMap[x][y] = p.val;
                        }
                        // 主要是为了确保短柱可以及时处理，至于为什么不一次性遍历全部是因为每个柱只能确保和它关联的区域的接水情况
                        priorityQueue.add(new Pair(heightMap[x][y], x, y));
                    }
                }
            }
        }
        return ans;
    }

    private boolean check(int x, int y) {
        return x >= 0 && y >= 0 && x < m && y < n && !marked[x][y];
    }

    private static class Pair {
        int val;
        int x, y;

        public Pair() {
        }

        public Pair(int val, int x, int y) {
            this.val = val;
            this.x = x;
            this.y = y;
        }

        @Override
        public String toString() {
            return "Pair{" +
                    "val=" + val +
                    ", x=" + x +
                    ", y=" + y +
                    '}';
        }
    }
}
```

#### [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

题目：

给定一个仅包含 `0` 和 `1` 、大小为 `rows x cols` 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/14/maximal.jpg)



解答：

终于回到我们熟悉的单调栈了，这题关键在于怎么想到把二维的图形压缩成一维图形。其实前面有一题，是求直方图中**最大矩阵面积**，我们可以想办法靠上去，所以会分层计算，首先以第一层为底，进行计算，至于说矩阵的高度，就是连续的1的长度，如果期间出现了0，说明发生了断层，前面的高度将不被囊括，此时高度清0，重新开始即可。



此时我们有代码：

``` Java
class Solution {
    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        int[] tmp = new int[matrix[0].length];
        int ans = 0;
        // 二维压扁成一维，这是最关键的地方，也是我没想到的地方
        for (int i = 0; i < matrix.length; ++ i) {
            for (int j = 0; j < matrix[0].length; ++ j) {
                if (matrix[i][j] == '1') {
                    tmp[j] ++;
                } else {
                    tmp[j] = 0;
                }
            }
            ans = Math.max(ans, largestRectangleArea(tmp));
        }
        return ans;
    }

    private final LinkedList<Pair> stack = new LinkedList<>();

    private void add1(Pair pair) {
        if (stack.isEmpty()) {
            stack.add(pair);
        } else {
            Pair first;
            while (!stack.isEmpty() && (first = stack.getFirst()) != null && pair.val < first.val) {
                first.next = pair.index;
                stack.removeFirst();
            }
            stack.addFirst(pair);
        }
    }

    private void add2(Pair pair) {
        if (stack.isEmpty()) {
            stack.add(pair);
        } else {
            Pair first;
            while (!stack.isEmpty() && (first = stack.getFirst()) != null && pair.val < first.val) {
                first.prev = pair.index;
                stack.removeFirst();
            }
            stack.addFirst(pair);
        }
    }

    private int largestRectangleArea(int[] heights) {
        int ans = 0;
        Pair[] pairs = new Pair[heights.length];
        for (int i = 0; i < heights.length; ++ i) {
            Pair pair = new Pair();
            pair.val = heights[i];
            pair.index = i;
            pair.prev = -1;
            pair.next = pairs.length;
            pairs[i] = pair;
            add1(pair);
        }
        stack.clear();
        for (int i = heights.length-1; i >= 0; -- i) {
            add2(pairs[i]);
        }
        for (int i = 0; i < pairs.length; ++ i) {
            // System.out.println(pairs[i]);
            ans = Math.max(ans, pairs[i].val * (pairs[i].next - pairs[i].prev - 1));
        }
        return ans;
    }

    private static class Pair {
        int val, index;
        // 找到后面小于它的，和前面小于它的
        int prev, next;

        @Override
        public String toString() {
            return "val: " + val + ",index: " + index + ", prev: " + prev + ", next: " + next;
        }
    }
}
```

#### [44. 通配符匹配](https://leetcode-cn.com/problems/wildcard-matching/)

题目：

给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。

'?' 可以匹配任何单个字符。
'*' 可以匹配任意字符串（包括空字符串）。
两个字符串完全匹配才算匹配成功。

说明:

s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。



解答：

又是通配符，又是两个字符串，直觉上DP。其实一般来说，**涉及到字符串匹配**的，包括回文串，上升子序列，上升子串，字符串匹配等，**基本都是DP解决**。



当然，像这道题，它就是硬编码，if-else硬匹配，枚举所有可能情况，其实涉及到匹配的基本都是这种套路，代码：

``` Java
class Solution {

    // 就硬做，没什么特殊的解法，枚举所有可能的状态转移方程
    public boolean isMatch(String s, String p) {
        char[] str1 = s.toCharArray();
        char[] str2 = p.toCharArray();
        boolean[][] dp = new boolean[str1.length+1][str2.length+1];
        if (str1.length == 0 && str2.length == 0) {
            return true;
        } else if (str2.length == 0) {
            return false;
        } else if (str1.length == 0) {
            for (char c : str2) {
                if (c != '*') {
                    return false;
                }
            }
            return true;
        }
        dp[0][0] = true;
        if (str2[0] == '*') {
            for (int i = 0; i < str1.length; ++ i) {
                dp[i+1][0] = true;
            }
        }
        for (int i = 0; i < str2.length; ++ i) {
            if (str2[i] == '*') {
                dp[0][i+1] = true;
            } else {
                break;
            }
        }
        for (int i = 1; i <= str1.length; ++ i) {
            for (int j = 1; j <= str2.length; ++ j) {
                if (str1[i-1] == str2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else if (str2[j-1] == '?') {
                    dp[i][j] = dp[i-1][j-1];
                } else if (str2[j-1] == '*') {
                    for (int k = i; k >= 0; -- k) {
                        dp[i][j] = dp[i][j] || dp[k][j-1];
                        if (dp[i][j]) {
                            break;
                        }
                    }
                } else {
                    dp[i][j] = false;
                }
            }
        }
        return dp[str1.length][str2.length];
    }
}
```

#### [135. 分发糖果](https://leetcode-cn.com/problems/candy/)

题目：

老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

每个孩子至少分配到 1 个糖果。
评分更高的孩子必须比他两侧的邻位孩子获得更多的糖果。
那么这样下来，老师至少需要准备多少颗糖果呢？



解答：

就硬模拟：

``` Java
class Solution {

    // 就硬做，没有什么特殊的解法，两个循环跑一遍，来进行数量的更新
    public int candy(int[] ratings) {
        int[] vals = new int[ratings.length];
        vals[0] = 1;
        for (int i = 1; i < ratings.length; ++i) {
            if (ratings[i] > ratings[i - 1]) {
                vals[i] = vals[i - 1] + 1;
            } else {
                vals[i] = 1;
            }
        }
        for (int i = ratings.length - 2; i >= 0; --i) {
            if (ratings[i] > ratings[i + 1]) {
                // 唯一需要注意的是这里，第二遍循环从后往前时不能单纯的+1，还要考虑第一遍的更新结果不能被覆盖
                vals[i] = Math.max(vals[i], vals[i + 1] + 1);
            }
        }
        int ans = 0;
        for (int a : vals) {
            ans += a;
            // System.out.print(a + " ");
        }
        // System.out.println();
        return ans;
    }
}
```

#### [149. 直线上最多的点数](https://leetcode-cn.com/problems/max-points-on-a-line/)

题目：

给你一个数组 `points` ，其中 `points[i] = [xi, yi]` 表示 **X-Y** 平面上的一个点。求最多有多少个点在同一条直线上。



解答：

就硬循环，之所以提这个，是因为我当初以为暴力一定超时，现在我拎出来说就是想说，有些题需要暴力模拟：

``` Java
class Solution {
    public int maxPoints(int[][] points) {
        int ans = 1;
        for (int i = 0; i < points.length; ++ i) {
            for (int j = i+1; j < points.length; ++ j) {
                int a = points[j][0] - points[i][0];
                int b = points[j][1] - points[i][1];
                int count = 2;
                for (int k = j+1; k < points.length; ++ k) {
                    int c = points[k][0] - points[i][0];
                    int d = points[k][1] - points[i][1];
                    if (a*d == b*c) {
                        ++ count;
                    }
                }
                ans = Math.max(ans, count);
            }
        }
        return ans;
    }
}
```

#### [887. 鸡蛋掉落](https://leetcode-cn.com/problems/super-egg-drop/)

题目：

给你 k 枚相同的鸡蛋，并可以使用一栋从第 1 层到第 n 层共有 n 层楼的建筑。

已知存在楼层 f ，满足 0 <= f <= n ，任何从 高于 f 的楼层落下的鸡蛋都会碎，从 f 楼层或比它低的楼层落下的鸡蛋都不会破。

每次操作，你可以取一枚没有碎的鸡蛋并把它从任一楼层 x 扔下（满足 1 <= x <= n）。如果鸡蛋碎了，你就不能再次使用它。如果某枚鸡蛋扔下后没有摔碎，则可以在之后的操作中 重复使用 这枚鸡蛋。

请你计算并返回要确定 f 确切的值 的 最小操作次数 是多少？



解答：

这题可太艹了，为啥呢？这是谷歌当年的面试题，答出来的人并不多。首先我们知道，楼层越多，次数肯定越多，鸡蛋越少，次数肯定也越多，所以次数和楼层成正比，和鸡蛋数成反比。其实第一时间大家很容易想到的是二分查找，为了优化一波，我们可以使用DP，直接定义dp\[i]\[j]: 鸡蛋数为i时，楼层为j所需要的最少次数；在处理转移方程时，其实是需要遍历的，遍历已经求得的子区间，因为我们大概只能知道鸡蛋破碎的位置在中间或以下，甚至中间以上，所以我们需要来一次遍历，当上下关系变化时，就是我们要找的位置了。



其实如果想要优化，还可以使用二分替代遍历，此时可能就可以通过，但是当初我仅仅是想到查找这一步就想了许久，没有继续下去，我们来看另一种方法。



我们必须重新定义dp数组。这也是比较困难的动态规划常见的方式——无法直接定义。此时定义成dp\[i]\[j]: 当鸡蛋数为i时，尝试次数为j所能到达的最高高度。此时有dp\[i]\[j] = dp\[i-1]\[j-1] + dp\[i]\[j-1] + 1;分为碎了和没碎的总和。

``` Java
class Solution {
    public int superEggDrop(int k, int n) {
        // 下面来看一种题解里看到的比较难以想到的方法
        // 核心在于更改dp定义，这也是hard题里常见的：不要用要求来定义DP，而要用答案来定义DP。
        // 这里的dp为：dp[鸡蛋个数][操作次数]=楼层高，就是在有i个鸡蛋的情况下，操作j次，最高能操作到几层。
        // 经过测试最多最多14次
        int[][] dp = new int[k+1][n+1];
        int j = 0;
        while (dp[k][j] < n) {
            ++ j;
            for (int i = 1; i <= k; ++ i) {
                // 想要看当前操作+鸡蛋数能到多少层，就要看操作数-1能到的楼层数，此时还分为鸡蛋碎了，和鸡蛋没碎两种情况。
                dp[i][j] = dp[i-1][j-1] + dp[i][j-1] + 1;
            }
        }
        return j;
   
```

#### [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)

题目：

给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。

字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）

题目数据保证答案符合 32 位带符号整数范围。



解答：

典型的字符串DP问题，直接定义即可，dp\[i]\[j]: 表示t的钱i个串在s的前j个串中出现的次数。此时我们有dp\[i]\[j] = dp\[i-1]\[j-1] + dp\[i]\[j-1]，这是在t\[i] == s\[j]的情况下，或者dp\[i]\[j] = dp\[i]\[j-1]这是在不等情况下。

``` Java
class Solution {
    public int numDistinct(String s, String t) {
        s = " " + s;
        t = " " + t;
        char[] str1 = t.toCharArray();
        char[] str2 = s.toCharArray();
        int[][] dp = new int[str1.length+1][str2.length+1];
        dp[0][0] = 1;
        for (int i = 1; i <= str1.length; ++ i) {
            for (int j = 1; j <= str2.length; ++ j) {
                if (str1[i-1] == str2[j-1]) {
                    dp[i][j] = dp[i-1][j-1]+dp[i][j-1];
                } else {
                    dp[i][j] = dp[i][j-1];
                }
            }
        }
        return dp[str1.length][str2.length];
    }
}
```

#### [1563. 石子游戏 V](https://leetcode-cn.com/problems/stone-game-v/)

题目：

几块石子 排成一行 ，每块石子都有一个关联值，关联值为整数，由数组 stoneValue 给出。

游戏中的每一轮：Alice 会将这行石子分成两个 非空行（即，左侧行和右侧行）；Bob 负责计算每一行的值，即此行中所有石子的值的总和。Bob 会丢弃值最大的行，Alice 的得分为剩下那行的值（每轮累加）。如果两行的值相等，Bob 让 Alice 决定丢弃哪一行。下一轮从剩下的那一行开始。

只 剩下一块石子 时，游戏结束。Alice 的分数最初为 0 。

返回 Alice 能够获得的最大分数 。



解答：

比较标准的内循环DP，同时需要进行区间切分，然后遍历找寻最值即可。

``` Java
class Solution {
    public int stoneGameV(int[] stoneValue) {
        long[][] dp = new long[stoneValue.length][stoneValue.length];
        long[] sum = new long[stoneValue.length+1];
        sum[0] = 0;
        sum[1] = stoneValue[0];
        // dp[0][0] = stoneValue[0];
        for (int i = 1; i < stoneValue.length; ++ i) {
            sum[i+1] = sum[i]+stoneValue[i];
            // dp[i][i] = stoneValue[i];
            dp[i-1][i] = Math.min(stoneValue[i-1], stoneValue[i]);
        }
        for (int j = 0; j < stoneValue.length; ++ j) {
            for (int i = j; i >= 0; -- i) {
                // l: [i, k-1]; r: [k, j];
                for (int k = j; k >= i+1; -- k) {
                    long left = sum[k]-sum[i];
                    long right = sum[j+1]-sum[k];
                    // System.out.println(left + ";" + right);
                    if (left < right) {
                        dp[i][j] = Math.max(dp[i][j], left + dp[i][k-1]);
                    } else if (left > right) {
                        dp[i][j] = Math.max(dp[i][j], right + dp[k][j]);
                    } else {
                        dp[i][j] = Math.max(dp[i][j], Math.max(dp[i][k-1], dp[k][j]) + left);
                    }
                }
                // System.out.println(i + "," + j + ": " + dp[i][j]);
            }
        }
        return (int) dp[0][stoneValue.length-1];
    }
}
```

#### [773. 滑动谜题](https://leetcode-cn.com/problems/sliding-puzzle/)

题目：

在一个 2 x 3 的板上（board）有 5 块砖瓦，用数字 1~5 来表示, 以及一块空缺用 0 来表示.

一次移动定义为选择 0 与一个相邻的数字（上下左右）进行交换.

最终当板 board 的结果是 [[1,2,3],[4,5,0]] 谜板被解开。

给出一个谜板的初始状态，返回最少可以通过多少次移动解开谜板，如果不能解开谜板，则返回 -1 。



解答：

其实很明显可以看出来时搜索，但是DFS可能不方便记录最短路径，在这里我们把石板每种状态记录成一个路径节点，计算的是起始状态到破解状态的最短的路径，所以可以使用Dijkstra算法，又因为此时权值为1，可以退化成BFS求解。

``` Java
class Solution {
    private int m = 0;

    private int n = 0;

    private final HashSet<Integer> marked = new HashSet<>();

    public int slidingPuzzle(int[][] board) {
        LinkedList<Pair> queue = new LinkedList<>();
        int[] X = new int[]{0, 1, 0, -1};
        int[] Y = new int[]{1, 0, -1, 0};
        m = board.length;
        n = board[0].length;
        Pair head = new Pair();
        head.arr = board;
        head.count = 0;
        queue.addLast(head);
        while (!queue.isEmpty()) {
            Pair first = queue.getFirst();
            queue.removeFirst();
            if (equals(first.arr)) {
                return first.count;
            } else {
                int a = 0, b = 0;
                for (int i = 0; i < board.length; ++ i) {
                    boolean flag = false;
                    for (int j = 0; j < board[0].length; ++ j) {
                        if (first.arr[i][j] == 0) {
                            a = i;
                            b = j;
                            flag = true;
                            break;
                        }
                    }
                    if (flag) {
                        break;
                    }
                }
                for (int i = 0; i < 4; ++ i) {
                    int x = a+X[i];
                    int y = b+Y[i];
                    if (checkBound(x, y)) {
                        int[][] tmpArr = swap(x, y, a, b, first);
                        if (tmpArr != null) {
                            Pair pair = new Pair();
                            pair.count = first.count+1;
                            pair.arr = tmpArr;
                            queue.addLast(pair);
                        }
                    }
                }
            }
        }
        return -1;
    }

    private int[][] swap(int x, int y, int a, int b, Pair pair) {
        int[][] arr = pair.arr;
        int tmp1 = arr[a][b];
        int tmp2 = arr[x][y];
        arr[a][b] = tmp2;
        arr[x][y] = tmp1;
        int key = getKey(arr);
        if (!marked.contains(key)) {
            marked.add(key);
            int[][] tmp = new int[m][n];
            for (int i = 0; i < m; ++ i) {
                System.arraycopy(arr[i], 0, tmp[i], 0, n);
            }
            arr[a][b] = tmp1;
            arr[x][y] = tmp2;
            return tmp;
        } else {
            arr[a][b] = tmp1;
            arr[x][y] = tmp2;
            return null;
        }
    }

    private int getKey(int[][] arr) {
        return arr[0][0] + arr[0][1]*6 + arr[0][2]*36 +
                arr[1][0]*216 + arr[1][1]*1296 + arr[1][2]*7776;
    }

    private boolean checkBound(int x, int y) {
        return x >= 0 && x < m && y >= 0 && y < n;
    }

    private boolean equals(int[][] array) {
        return array[0][0] == 1 && array[0][1] == 2 && array[0][2] == 3 &&
                array[1][0] == 4 && array[1][1] == 5 && array[1][2] == 0;
    }

    private static class Pair {
        private int count;
        private int[][] arr;

        @Override
        public String toString() {
            return "arr: " + Arrays.toString(arr[0]) + "&" + Arrays.toString(arr[1]) + ", " + "count: " + count;
        }
    }
}
```

#### [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

题目：

给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。



**示例 1:**

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram.jpg)



解答：

这是一个寻找写一个小于当前值的题目，所以我们要使用单调栈：

``` Java
class Solution {
    
    private final LinkedList<Pair> stack = new LinkedList<>();

    private void add1(Pair pair) {
        Pair first;
        while (!stack.isEmpty() && (first = stack.getFirst()) != null && pair.val < first.val) {
            first.next = pair.index;
            stack.removeFirst();
        }
        stack.addFirst(pair);
    }

    private void add2(Pair pair) {
        if (stack.isEmpty()) {
            stack.add(pair);
        } else {
            Pair first;
            while (!stack.isEmpty() && (first = stack.getFirst()) != null && pair.val < first.val) {
                first.prev = pair.index;
                stack.removeFirst();
            }
            stack.addFirst(pair);
        }
    }

    public int largestRectangleArea(int[] heights) {
        int ans = 0;
        Pair[] pairs = new Pair[heights.length];
        for (int i = 0; i < heights.length; ++ i) {
            Pair pair = new Pair();
            pair.val = heights[i];
            pair.index = i;
            pair.prev = -1;
            pair.next = pairs.length;
            pairs[i] = pair;
            add1(pair);
        }
        stack.clear();
        for (int i = heights.length-1; i >= 0; -- i) {
            add2(pairs[i]);
        }
        for (int i = 0; i < pairs.length; ++ i) {
            // System.out.println(pairs[i]);
            ans = Math.max(ans, pairs[i].val * (pairs[i].next - pairs[i].prev - 1));
        }
        return ans;
    }

    private static class Pair {
        int val, index;
        int prev, next;

        @Override
        public String toString() {
            return "val: " + val + ",index: " + index + ", prev: " + prev + ", next: " + next;
        }
    }
}
```

#### [295. 数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)

题目：

中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。

例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。



解答：

动态数据，动态数组的中位数问题，中位数可以使用优先队列，动态求最值可以使用优先队列，所以我们维护两个优先队列，一个是全部小于中位数的，一个是全部大于中位数的，然后取两者极值即可。



关键在于如何因为数据的动态添加而维护两个优先队列，确保他们的大小最多相差1，以及始终拥有一个全部大于另一个的特性：

``` Java
class MedianFinder {

        private final PriorityQueue<Integer> min = new PriorityQueue<>();

        private final PriorityQueue<Integer> max = new PriorityQueue<>((p1, p2) -> -1 * Integer.compare(p1, p2));

        /** initialize your data structure here. */
        public MedianFinder() {
            ;
        }

  			// 对于双优先队列的维护，只能通过add函数处理，不然维护成本太高了。
        public void addNum(int num) {
            max.add(num);
            // 始终保证min里的元素>=max里的元素
            min.add(max.poll());
            // 维持两个队列大小，防止差距过大
            if (min.size() > max.size()) {
                max.add(min.poll());
            }
        }

        public double findMedian() {
            if (min.size() == max.size()) {
                return (min.peek() + max.peek()) * 1.0 / 2;
            } else {
                return max.peek();
            }
        }
    }

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
```

#### [765. 情侣牵手](https://leetcode-cn.com/problems/couples-holding-hands/)

N 对情侣坐在连续排列的 2N 个座位上，想要牵到对方的手。 计算最少交换座位的次数，以便每对情侣可以并肩坐在一起。 一次交换可选择任意两人，让他们站起来交换座位。

人和座位用 0 到 2N-1 的整数表示，情侣们按顺序编号，第一对是 (0, 1)，第二对是 (2, 3)，以此类推，最后一对是 (2N-2, 2N-1)。

这些情侣的初始座位  row[i] 是由最初始坐在第 i 个座位上的人决定的。



解答：

这道题其实需要理解一件事，就是情侣之间处于一个并查集，同时处于一个并查集内的情侣，我们认为他们的座位是排好的，所以最后只要统计不同并查集数量即可。



``` Java
class Solution {
    private int[] parent;

    public int minSwapsCouples(int[] row) {
        parent = new int[row.length];
        for (int i = 0; i < row.length; ++i) {
            parent[i] = i % 2 == 0 ? i : i - 1;
        }
        int sum = 0;
        for (int i = 0; i < row.length; i += 2) {
            int a = find(row[i]);
            int b = find(row[i+1]);
            if (a != b) {
                ++ sum;
                // 表示进行交换
                merge(a, b);
            }
        }
        return sum;
    }

    private int find(int i) {
        int index = i;
        while (parent[i] != i) {
            i = parent[i];
        }
        parent[index] = i;
        return i;
    }

    private void merge(int a, int b) {
        parent[find(a)] = find(b);
    }
}
```

#### [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

题目：

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

 

注意：

对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
如果 s 中存在这样的子串，我们保证它是唯一的答案。



解答：

字符串，区间问题，铁定是...哈哈哈哈不是DP，是滑动窗口。因为t可能包含重复的，所以我们需要进行记录每个字符的个数，同时为了方便统计是否已经全部覆盖，我们还需要记录当前已经覆盖的总数，再之，我们还需要记录s中区间内的字符是否贡献了自己的价值，也即，去除重复记录。这要求我们在减少count时记得先判断这个字符是否还剩下没有被覆盖的。

``` Java
class Solution {
    public String minWindow(String s, String t) {
        char[] str1 = s.toCharArray();
        char[] str2 = t.toCharArray();
        int[] rcd = new int[256];
        boolean[] marked = new boolean[256];
        int count = str2.length;
        for (char c : str2) {
            marked[c] = true;
            rcd[c]++;
        }
        int minLen = Integer.MAX_VALUE;
        int minL = 0, minR = 0;
        int l = 0, r = 0;
        boolean run = false;
        while (r < str1.length && l <= r) {
            if (marked[str1[r]]) {
                if (rcd[str1[r]] > 0) {
                    --count;
                }
                --rcd[str1[r]];
            }
            if (count == 0) {
                run = true;
                while (count == 0) {
                    if (marked[str1[l]]) {
                        if (rcd[str1[l]] >= 0) {
                            ++count;
                        }
                        ++rcd[str1[l]];
                    }
                    ++l;
                }
                if (minLen > (r-(l-1))+1) {
                    minLen = r-(l-1)+1;
                    minL = l-1;
                    minR = r;
                }
            }
            ++r;
        }
        if (!run) {
            return "";
        }
        return new String(str1, minL, minR-minL+1);
    }
}
```

#### [480. 滑动窗口中位数](https://leetcode-cn.com/problems/sliding-window-median/)

题目：

中位数是有序序列最中间的那个数。如果序列的长度是偶数，则没有最中间的数；此时中位数是最中间的两个数的平均数。

例如：

[2,3,4]，中位数是 3
[2,3]，中位数是 (2 + 3) / 2 = 2.5
给你一个数组 nums，有一个长度为 k 的窗口从最左端滑动到最右端。窗口中有 k 个数，每次窗口向右移动 1 位。你的任务是找出每次窗口移动后得到的新窗口中元素的中位数，并输出由它们组成的数组。



解答：

这题既然出现了滑动窗口，中位数，那我们肯定就要上优先队列，还是两个，来进行中位数查找。但是这里有个问题，已经滑过的数，怎么剔除？优先队列可不支持指定数据删除。其实如果你很牛的话，可以写一个红黑树来完成这个步骤，但是我想如果你能写红黑树，那你肯定也看不起这个文章了。所以我们要想个办法，可以找到边界外的元素并删除。



不卖关子了，我们这里使用延迟删除策略，其他的可能会超时。什么是延迟删除呢？我们前面说滑动窗口的最大值已经提到了，现在来看更加复杂的。首先判断边界元素在哪个优先队列。然后把这个优先队列的平衡值+1，当我们进行平衡两个优先队列时，会把平衡值算进去，然后会检测队首元素是否需要剔除，然后根据前面数据流的中位数做法，获取中位数。



通过延迟删除，可以避免重复循环，进而实现O(nlogn)的时间复杂度：

``` Java
class Solution {

    private PriorityQueue<Pair> min = new PriorityQueue<>((p1, p2) -> {
        if (p1.val == p2.val) {
            return Integer.compare(p1.index, p2.index);
        } else {
            return Integer.compare(p1.val, p2.val);
        }
    });

    private PriorityQueue<Pair> max = new PriorityQueue<>((p1, p2) -> {
        if (p1.val == p2.val) {
            return Integer.compare(p1.index, p2.index);
        } else {
            return -1 * Integer.compare(p1.val, p2.val);
        }
    });

    private int inMin = 0;

    private int inMax = 0;

    public double[] medianSlidingWindow(int[] nums, int k) {
        double[] ans = new double[nums.length - k + 1];
        int index = 0;
        // 预处理
        for (int i = 0; i < k; ++i) {
            max.add(new Pair(nums[i], i));
            min.add(max.poll());
            if (min.size() > max.size()) {
                max.add(min.poll());
            }
        }
        ans[index++] = max.size() == min.size() ? (max.peek().val * 1.0 + min.peek().val * 1.0) / 2.0 : max.peek().val * 1.0;
        for (int i = k; i < nums.length; ++ i) {
            // 添加元素
            add(nums[i], i, nums[i-k], i-k);
            // 处理中位数
            if ((max.size()-inMax) == (min.size()-inMin)) {
                ans[index++] = (max.peek().val * 1.0 + min.peek().val * 1.0) / 2.0;
            } else {
                ans[index++] = max.peek().val * 1.0;
            }
        }
        return ans;
    }

    private PriorityQueue<Pair> print(PriorityQueue<Pair> priorityQueue) {
        PriorityQueue<Pair> p = new PriorityQueue<>(priorityQueue.comparator());
        while (!priorityQueue.isEmpty()) {
            System.out.printf("val: %011d + idx: %02d%n", priorityQueue.peek().val, priorityQueue.peek().index);
            p.add(priorityQueue.poll());
        }
        return p;
    }

    private void add(int val, int index, int excludedValue, int excludedIndex) {
        // 边界元素在哪个优先队列中
        if (inWhich(excludedValue, excludedIndex)) {
            ++inMax;
        } else {
            ++inMin;
        }
        max.add(new Pair(val, index));
        // 剔除边界元素
        while (!max.isEmpty() && max.peek().index <= excludedIndex) {
            --inMax;
            max.poll();
        }
        // 保持平衡
        min.add(max.poll());
        // 剔除边界元素
        while (!min.isEmpty() && min.peek().index <= excludedIndex) {
            --inMin;
            min.poll();
        }
        // 依旧保持平衡，这里算上了需要剔除的元素数量，所以是平衡的
        if ((min.size()-inMin) > (max.size()-inMax)) {
            max.add(min.poll());
        }
        // 下面两个循环依旧是剔除边界元素
        while (!max.isEmpty() && max.peek().index <= excludedIndex) {
            --inMax;
            max.poll();
        }
        while (!min.isEmpty() && min.peek().index <= excludedIndex) {
            --inMin;
            min.poll();
        }
    }

    // True：在max中
    private boolean inWhich(int val, int index) {
        if (min.isEmpty()) {
            return true;
        }
        if (max.isEmpty()) {
            return false;
        }
        if (val < max.peek().val) {
            return true;
        }
        if (val > min.peek().val) {
            return false;
        }
        if (val == max.peek().val && index == max.peek().index) {
            return true;
        }
        if (val == min.peek().val && index == min.peek().index) {
            return false;
        }
        return false;
    }

    private static class Pair {
        private int val;
        private int index;

        Pair() {
        }

        Pair(int val, int index) {
            this.val = val;
            this.index = index;
        }
    }
}
```

#### [440. 字典序的第K小数字](https://leetcode-cn.com/problems/k-th-smallest-in-lexicographical-order/)

题目：

给定整数 `n` 和 `k`，找到 `1` 到 `n` 中字典序第 `k` 小的数字。

注意：1 ≤ k ≤ n ≤ 109。



解答：

一般某题简短精炼，那八成是很难的。原本使用找规律没做出来，后来发现需要建树，但是这不是简单的字典树，虽然也可以，但是我们需要建立一个虚拟的树，来模拟这个过程。



待做...

#### [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/)

题目：

给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 你 不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。



解答：

很经典的DFS问题，直接DFS跑一遍即可。

``` Java
class Solution {
    public int longestIncreasingPath(int[][] matrix) {
        map = matrix;
        rcd = new int[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; ++ i) {
            for (int j = 0; j < matrix[0].length; ++ j) {
                int tmp = dfs(i, j);
                ans = Math.max(ans, tmp);
            }
        }
        return ans;
    }

    private int[][] rcd;
    
    private int[][] map;

    private int ans = -1;
    
    private final int[] X = new int[] {0, 1, 0, -1};
    
    private final int[] Y = new int[] {1, 0, -1, 0};
    
    private boolean checkBound(int x, int y) {
        return x >= 0 && x < rcd.length && y >= 0 && y < rcd[0].length;
    }

    private int dfs(int x, int y) {
        if (!checkBound(x, y)) {
            return 0;
        }
        if (rcd[x][y] == 0) {
            rcd[x][y] = 1;
        }
        int tmp = 1;
        for (int i = 0; i < 4; ++ i) {
            int a = x + X[i];
            int b = y + Y[i];
            if (!checkBound(a, b) || map[x][y] >= map[a][b])
                continue;
            if (rcd[a][b] != 0) {
                tmp = Math.max(tmp, rcd[a][b]+1);
            } else {
                tmp = Math.max(tmp, dfs(a, b)+1);
            }
        }
        rcd[x][y] = tmp;
        return tmp;
    }
}
```

#### [51. N 皇后](https://leetcode-cn.com/problems/n-queens/)

题目：

n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。



解答：

N皇后问题是典型的DFS回溯问题，我们这里直接给出代码好了：

``` Java
class Solution {
    
    public List<List<String>> solveNQueens(int n) {
        bound = n;
        dfs(0, new LinkedList<>());
        return ans;
    }

    private int bound;

    private final int[][] marked = new int[10][10];

    private final LinkedList<List<String>> ans = new LinkedList<>();

    private void set(int x, int y) {
        int a = x, b = y;
        while (a < bound) {
            marked[a][y] ++;
            ++ a;
        }
        a = x;
        while (a < bound && b < bound) {
            marked[a][b] ++;
            ++ a;
            ++ b;
        }
        a = x;
        b = y;
        while (a < bound && b >= 0) {
            marked[a][b] ++;
            ++ a;
            -- b;
        }
    }

    private void unset(int x, int y) {
        int a = x, b = y;
        while (a < bound) {
            marked[a][y] --;
            ++ a;
        }
        a = x;
        while (a < bound && b < bound) {
            marked[a][b] --;
            ++ a;
            ++ b;
        }
        a = x;
        b = y;
        while (a < bound && b >= 0) {
            marked[a][b] --;
            ++ a;
            -- b;
        }
    }

    private void dfs(int x, LinkedList<int[]> linkedList) {
        if (x == bound) {
            LinkedList<String> list = new LinkedList<>();
            for (int[] a : linkedList) {
                StringBuilder stringBuilder = new StringBuilder();
                for (int i = 0; i < a[1]; ++ i) {
                    stringBuilder.append(".");
                }
                stringBuilder.append("Q");
                for (int i = a[1]+1; i < bound; ++ i) {
                    stringBuilder.append(".");
                }
                list.addLast(stringBuilder.toString());
            }
            ans.addLast(list);
            return ;
        }
        for (int j = 0; j < bound; ++ j) {
            if (marked[x][j] == 0) {
                int[] tmp = new int[2];
                tmp[0] = x;
                tmp[1] = j;
                set(x, j);
                linkedList.addLast(tmp);
                dfs(x+1, linkedList);
                linkedList.removeLast();
                unset(x, j);
            }
        }
    }
}
```

#### [321. 拼接最大数](https://leetcode-cn.com/problems/create-maximum-number/)

题目：

给定长度分别为 m 和 n 的两个数组，其元素由 0-9 构成，表示两个自然数各位上的数字。现在从这两个数组中选出 k (k <= m + n) 个数字拼接成一个新的数，要求从同一个数组中取出的数字保持其在原数组中的相对顺序。

求满足该条件的最大数。结果返回一个表示该最大数的长度为 k 的数组。



解答：

我们可以考虑使用区间最优来求解整体最优，因为只有两个数组，我们不妨试试在数组1取a个数，数组2取m-a个数的分别最大值，然后拼接（合并两个有序数组），所以a的变化范围是0-(max(m, arr2.len))。接下来就是怎么求救当需要a个元素时，数组的最优解，那肯定是单调栈，因为我们可以根据数字大小排序，然后找到后面序号大于它的即可，找到后面第一个大于/小于问题都是单调栈问题。当然，此时我们必须注意剩下的元素个数能不能满足a，如果不能就不能再找下去了，直接全部添加。



代码：

``` Java
class Solution {
    private final LinkedList<Integer> stack1 = new LinkedList<>();

    private final LinkedList<Integer> stack2 = new LinkedList<>();

    private int[] nums1;

    private int[] nums2;

    private int[] nums;

    private boolean put1(int val, int remain, int must) {
        Integer a;
        while ((a = stack1.peekLast()) != null && val > a) {
            stack1.removeLast();
            if (remain + stack1.size() + 1 < must) {
                stack1.addLast(a);
                stack1.addLast(val);
                return false;
            }
        }
        stack1.addLast(val);
        return true;
    }

    private boolean put2(int val, int remain, int must) {
        Integer a;
        while ((a = stack2.peekLast()) != null && val > a) {
            stack2.removeLast();
            if (remain + stack2.size() + 1 < must) {
                stack2.addLast(a);
                stack2.addLast(val);
                return false;
            }
        }
        stack2.addLast(val);
        return true;
    }

    private void merge(int len1, int len2, int from1, int from2) {
        int[] tmp1 = new int[len1];
        int[] tmp2 = new int[len2];
        int idx1 = 0, idx2 = 0;
//        for (int a : stack1) {
//            System.out.print(a + " ");
//        }
//        System.out.println();
//        for (int a : stack2) {
//            System.out.print(a + " ");
//        }
//        System.out.println();
        while (len1 > 0 && !stack1.isEmpty()) {
            tmp1[idx1++] = stack1.getFirst();
            stack1.removeFirst();
            -- len1;
        }
        while (len1 > 0) {
            tmp1[idx1++] = nums1[from1++];
            -- len1;
        }
        while (len2 > 0 && !stack2.isEmpty()) {
            tmp2[idx2++] = stack2.getFirst();
            stack2.removeFirst();
            -- len2;
        }
        while (len2 > 0) {
            tmp2[idx2++] = nums2[from2++];
            -- len2;
        }
        compareAndSet(tmp1, tmp2);
    }

    private void compareAndSet(int[] nums1, int[] nums2) {
        int idx1 = 0, idx2 = 0;
        int idx = 0;
        int[] tmp = new int[nums.length];
        while (idx1 < nums1.length && idx2 < nums2.length) {
            if (nums1[idx1] > nums2[idx2]) {
                tmp[idx++] = nums1[idx1++];
            } else if (nums1[idx1] < nums2[idx2]) {
                tmp[idx++] = nums2[idx2++];
            } else {
                int tmp1 = idx1+1;
                int tmp2 = idx2+1;
                while (tmp1 < nums1.length && tmp2 < nums2.length) {
                    if (nums1[tmp1] > nums2[tmp2]) {
                        tmp[idx++] = nums1[idx1++];
                        break;
                    } else if (nums1[tmp1] < nums2[tmp2]) {
                        tmp[idx++] = nums2[idx2++];
                        break;
                    } else {
                        ++tmp1;
                        ++tmp2;
                    }
                }
                if (tmp1 == nums1.length) {
                    tmp[idx++] = nums2[idx2++];
                }
                if (tmp2 == nums2.length) {
                    tmp[idx++] = nums1[idx1++];
                }
            }
        }
        while (idx1 < nums1.length) {
            tmp[idx++] = nums1[idx1++];
        }
        while (idx2 < nums2.length) {
            tmp[idx++] = nums2[idx2++];
        }
        boolean flag = false;
        for (int i = 0; i < nums.length; ++i) {
            if (tmp[i] < nums[i]) {
                flag = true;
                break;
            } else if (tmp[i] > nums[i]) {
                break;
            }
        }
        if (!flag) {
            nums = tmp;
        }
    }

    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        this.nums1 = nums1;
        this.nums2 = nums2;
        nums = new int[k];
        int a = 0, b = k;
        // 进行范围圈定，防止切分时越界，即防止每个数组取到大于它元素个数的拼接方案
        while (b > nums2.length) {
            ++ a;
            -- b;
        }
        // 进行切分，即数组1取1-a个元素，然后得到最佳方案进行对比
        while (a <= nums1.length && b >= 0) {
            int from1 = nums1.length - 1, from2 = nums2.length - 1;
            for (int i = 0; i < nums1.length; ++ i) {
                // 进行单调栈添加，如果添加失败，说明剩余的元素不够了，直接跳出
                if (!put1(nums1[i], nums1.length - i - 1, a)) {
                    from1 = i+1;
                    break;
                }
            }
            for (int i = 0; i < nums2.length; ++ i) {
                // 同上
                if (!put2(nums2[i], nums2.length - i - 1, b)) {
                    from2 = i+1;
                    break;
                }
            }
            // 合并两个有序数组，并进行比较，找到所有组合中最大的那个
            merge(a, b, from1, from2);
            stack1.clear();
            stack2.clear();
            // System.out.println(Arrays.toString(nums));
            ++ a;
            -- b;
        }
        return nums;
    }
}
```

#### [1713. 得到子序列的最少操作次数](https://leetcode-cn.com/problems/minimum-operations-to-make-a-subsequence/)

题目：

给你一个数组 target ，包含若干 互不相同 的整数，以及另一个整数数组 arr ，arr 可能 包含重复元素。

每一次操作中，你可以在 arr 的任意位置插入任一整数。比方说，如果 arr = [1,4,1,2] ，那么你可以在中间添加 3 得到 [1,4,3,1,2] 。你可以在数组最开始或最后面添加整数。

请你返回 最少 操作次数，使得 target 成为 arr 的一个子序列。

一个数组的 子序列 指的是删除原数组的某些元素（可能一个元素都不删除），同时不改变其余元素的相对顺序得到的数组。比方说，[2,7,4] 是 [4,2,3,7,2,1,4] 的子序列（加粗元素），但 [2,4,2] 不是子序列。



解答：

这题本质是最长上升子序列的变种题，以target的元素排列为顺序，求arr的最长上升子序列。所以这里讲一下最长上升子序列：

``` Java
class Solution {

  	// 这个解法是比较通用的，通过贪心思想，进行替换，使用后面更小的元素进行替换。
  	// 我们有一点需要注意，这个解法更像是已知方案下推导出来的。
    public int lengthOfLIS(int[] nums) {
        int len = 1, n = nums.length;
        if (n == 0) {
            return 0;
        }
        int[] d = new int[n + 1];
        d[len] = nums[0];
        for (int i = 1; i < n; ++i) {
            if (nums[i] > d[len]) {
                d[++len] = nums[i];
            } else {
                int l = 1, r = len, pos = 0; // 如果找不到说明所有的数都比 nums[i] 大，此时要更新 d[1]，所以这里将 pos 设为 0
              	// 这里使用二分查找进行找到从后往前找第一个小于当前元素的下一个元素的位置
                while (l <= r) {
                    int mid = (l + r) >> 1;
                    if (d[mid] < nums[i]) {
                        pos = mid;
                        l = mid + 1;
                    } else {
                        r = mid - 1;
                    }
                }
                d[pos + 1] = nums[i];
            }
        }
        return d[len] == Integer.MAX_VALUE ? len-1 : len;
    }
  	
  	// 这个解法时间复杂度肯定没有上面那个好，但是比较通俗易懂
  	// 它是通过不断更新后面的长度来实现的
  	public int lengthOfLis(int[] nums) {
      	int[] dp = new int[nums.length+1];
      	for (int i = 0; i < nums.length; ++ i) {
          	for (int j = i + 1; j < nums.length; ++ j) {
              	// 这里是关键
              	if (nums[j] > nums[i]) {
                  	dp[j] = Math.max(dp[j], dp[i] + 1);
                }
            }
        }
      	int ans = 0;
      	for (int a : dp) {
          	ans = Math.max(ans, a);
        }
      	return ans;
    }
}
```

#### [132. 分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)

题目：

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是回文。

返回符合要求的 **最少分割次数** 。



解答：

首先，我们通过DP记录所有的回文串的位置，即通过dp\[i]\[j]是true还是false来进行记录i-j是不是回文串。其次，进行统计，因为切分点一定是回文串的位置，也就是i或j的位置，所以内循环切分比较即可。

``` Java
class Solution {
    private boolean[][] dp;

    private String string;

    public int minCut(String s) {
        char[] str = s.toCharArray();
        string = s;
        dp = new boolean[str.length][str.length];
        for (int i = 0; i < dp.length; ++ i) {
            dp[i][i] = true;
        }
        for (int i = 0; i < str.length; ++ i) {
            if (i-1 >= 0 && str[i] == str[i-1]) {
                dp[i-1][i] = true;
            }
            for (int j = i-2; j >= 0; -- j) {
                dp[j][i] = str[i] == str[j] && dp[j + 1][i - 1];
            }
        }
        int rcd[] = new int[str.length];
        for (int i = 0; i < rcd.length; ++ i) {
            rcd[i] = i;
            if (dp[0][i]) {
                rcd[i] = 0;
            } else {
                for (int j = i; j >= 1; -- j) {
                    // 寻找分割点，所有的回文子串的位置都是分割点
                    if (dp[j][i]) {
                        rcd[i] = Math.min(rcd[i], rcd[j-1] + 1);
                    }
                }
            }
        }
        return rcd[str.length - 1];
    }
}
```

#### [987. 二叉树的垂序遍历](https://leetcode-cn.com/problems/vertical-order-traversal-of-a-binary-tree/)

题目：

给你二叉树的根结点 root ，请你设计算法计算二叉树的 垂序遍历 序列。

对位于 (row, col) 的每个结点而言，其左右子结点分别位于 (row + 1, col - 1) 和 (row + 1, col + 1) 。树的根结点位于 (0, 0) 。

二叉树的 垂序遍历 从最左边的列开始直到最右边的列结束，按列索引每一列上的所有结点，形成一个按出现位置从上到下排序的有序列表。如果同行同列上有多个结点，则按结点的值从小到大进行排序。

返回二叉树的 垂序遍历 序列。



**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/01/29/vtree1.jpg)



解答：

一道比较常规的模拟题，需要我们好好设计比较原则，然后进行排序。

``` Java
class Solution {
    private int size = 0;

    private int count = 0;

    private Pair[] pairs;

    public List<List<Integer>> verticalTraversal(TreeNode root) {
        TreeNode0 root0 = f(root, 0, 0);
        pairs = new Pair[size];
        ff(root0);
        Arrays.sort(pairs, new Comparator<Pair>() {
            @Override
            public int compare(Pair o1, Pair o2) {
                if (o1.index == o2.index) {
                    if (o1.depth == o2.depth) {
                        return Integer.compare(o1.val, o2.val);
                    } else {
                        return Integer.compare(o1.depth, o2.depth);
                    }
                } else {
                    return Integer.compare(o1.index, o2.index);
                }
            }
        });
        List<List<Integer>> ans = new LinkedList<>();
        LinkedList<Integer> tmp = null;
        int a = pairs[0].index - 1;
        for (Pair pair : pairs) {
            if (pair.index != a) {
                tmp = new LinkedList<>();
                a = pair.index;
                ans.add(tmp);
            }
            tmp.addLast(pair.val);
        }
        return ans;
    }

    private void ff(TreeNode0 treeNode0) {
        if (treeNode0 == null) {
            return ;
        }
        pairs[count] = new Pair();
        pairs[count].val = treeNode0.val;
        pairs[count].index = treeNode0.index;
        pairs[count].depth = treeNode0.depth;
        ++ count;
        ff(treeNode0.left);
        ff(treeNode0.right);
    }

    private TreeNode0 f(TreeNode treeNode, int index, int depth) {
        if (treeNode  == null) {
            return null;
        }
        ++ size;
        TreeNode0 node = new TreeNode0();
        node.index = index;
        node.val = treeNode.val;
        node.depth = depth;
        node.left = f(treeNode.left, index-1, depth+1);
        node.right = f(treeNode.right, index+1, depth+1);
        return node;
    }

    private static class TreeNode0 {
        int val;
        int index;
        int depth;
        TreeNode0 left;
        TreeNode0 right;
    }

    private static class Pair {
        int val;
        int index;
        int depth;
    }
}
```

#### [123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

题目：

给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。



解答：

这题我说实话，我刚看到是很蒙的，后来看评论区，看题解得到了启发，就是我们可以把买卖分开，这样依次计算，但是我们必须明白一件事，就是**买卖之间是有关联的**。比方说我第二次买肯定依赖于第一次卖得到的钱，第一次卖肯定依赖于第一次买花的钱，第二次卖肯定依赖于第二次买花的钱。所以我们得到了如下的转移方程。如果不用DP还真的不好求解，因为这买卖之间有关系依赖，且存在状态转移。

``` Java
class Solution {
    public int maxProfit(int[] nums) {
        // 在第i天第一次卖所能获得的最大收益
        int[] sell1 = new int[nums.length+2];
        // 在第i天第二次卖所能获得的最大收益
        int[] sell2 = new int[nums.length+2];
        // 在第i天第一次买所能获得的最大收益
        int[] buy1 = new int[nums.length+2];
        // 在第i天第二次买所能获得的最大收益
        int[] buy2 = new int[nums.length+2];
        sell1[0] = -nums[0];
        // 此时无本金，买入手里的钱肯定是负的
        buy1[0] = -nums[0];
        sell2[0] = -nums[0];
        // 此时无本金，买入手里的钱肯定是负的
        buy2[0] = -nums[0];
        for (int i = 0; i < nums.length; ++ i) {
            // 分为不卖，和卖了同时赚取这一天股票的价格的钱，因为卖之前必须先买，所以需要算上第一次买的钱(负的)
            sell1[i+1] = Math.max(sell1[i], buy1[i]+nums[i]);
            // 同上
            sell2[i+1] = Math.max(sell2[i], buy2[i]+nums[i]);
            // 第二次买分为不买，和买入当日股票所花的钱，所以要减去，第二次买肯定要求之前卖过，之前第一次卖的钱就要减去今天的股票价格得到此时还剩的钱
            buy2[i+1] = Math.max(buy2[i], sell1[i]-nums[i]);
            // 第一次买肯定要贴钱，所以是0-nums[i]，以及不买的钱buy1[i]进行比较
            buy1[i+1] = Math.max(buy1[i], -nums[i]);
        }
        return Math.max(sell1[nums.length], sell2[nums.length]);
    }
}
```

#### [188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

题目：

给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。



解答：

这题和上一题最大的不同，在于要求使用给出的交易次数，所以我们可以推理得到这题的解法应该是再加一个循环k来进行每次交易的最值求解，事实证明我们猜的是对的。

``` Java
class Solution {
    public int maxProfit(int k, int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int[][] sell = new int[nums.length+1][k+1];
        int[][] buy = new int[nums.length+1][k+1];
      	// 初始化
        for (int i = 0; i < k; ++ i) {
            sell[0][i] = -nums[0];
            buy[0][i] = -nums[0];
        }
        for (int i = 0; i < nums.length; ++ i) {
          	// 对于第一次买卖需要特殊处理一下
            sell[i+1][0] = Math.max(sell[i][0], buy[i][0]+nums[i]);
            buy[i+1][0] = Math.max(buy[i][0], -nums[i]);
            for (int j = 1; j < k; ++ j) {
                sell[i+1][j] = Math.max(sell[i][j], buy[i][j]+nums[i]);
                buy[i+1][j] = Math.max(buy[i][j], sell[i][j-1]-nums[i]);
            }
        }
        int max = 0;
        for (int i = 0; i < k; ++ i) {
            max = Math.max(max, sell[nums.length][i]);
        }
        return max;
    }
}
```

#### [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)

题目：

给你一个未排序的整数数组 `nums` ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 `O(n)` 并且只使用常数级别额外空间的解决方案。



解答：

这题其实有点脑筋急转弯的思想，一般来说脑筋急转弯大多出现在normal里。既然找第一个缺失的正数，那我们可以假设在第一个缺失的正数之前的数组，肯定是nums\[i] >= 1且单调递增，所以我们可以设置一个期待值，一旦nums\[i]出现了跨越，比如从k => k+2，那我们就可以知道缺失了k+1。



当然我的题解不太好：

``` Java
class Solution {
    public int firstMissingPositive(int[] nums) {
        HashSet<Integer> set = new HashSet<>(nums.length);
        for (int a : nums) {
            set.add(a);
        }
        for (int i = 1; i <= nums.length; ++ i) {
            if (!set.contains(i)) {
                return i;
            }
        }
        return nums.length+1;
    }
}
```

#### [60. 排列序列](https://leetcode-cn.com/problems/permutation-sequence/)

题目：

给出集合 [1,2,3,...,n]，其所有元素共有 n! 种排列。

按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：

"123"
"132"
"213"
"231"
"312"
"321"
给定 n 和 k，返回第 k 个排列。



解答：

全排列问题，很容易想到使用回溯求解，回溯时记得记录一下这是第几轮，然后等于k时直接设置答案即可。

``` Java
class Solution {

    public String getPermutation(int n, int k) {
        this.k = k;
        ff(new int[n], 0);
        return ans;
    }

    private boolean[] marked = new boolean[10];

    private int k;

    private String ans = "";

    private void ff(int[] nums, int index) {
        if (index >= nums.length) {
            -- k;
            if (k == 0) {
                for (int a : nums) {
                    ans += a;
                }
            }
            return ;
        }
        for (int i = 0; i < nums.length; ++ i) {
            if (!marked[i+1]) {
                nums[index] = i+1;
                marked[i+1] = true;
                ff(nums, index+1);
                marked[i+1] = false;
            }
        }
    }
}
```

#### [30. 串联所有单词的子串](https://leetcode-cn.com/problems/substring-with-concatenation-of-all-words/)

题目：

给定一个字符串 s 和一些 长度相同 的单词 words 。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。

注意子串要与 words 中的单词完全匹配，中间不能有其他字符 ，但不需要考虑 words 中单词串联的顺序。



解答：

是一道很典型的滑动窗口的题目，因为涉及到找区间存在性，那一般就是滑动窗口，这里我们使用了和之前一样的做法：记录count的同时通过记录重复元素来保证count贡献值始终是对的，所以代码如下：

``` Java
class Solution {

    public List<Integer> findSubstring(String s, String[] words) {
        HashMap<String, Integer> marked = new HashMap<>();
        for (String str : words) {
            marked.merge(str, 1, Integer::sum);
        }
        int len1 = words[0].length();
        for (int i = 0; i <= s.length() - len1; ++ i) {
            idx2Str.put(i, s.substring(i, i + len1));
        }
        int l = 0, r = 0;
        int count = words.length;
        ArrayList<Integer> ans = new ArrayList<>();
        for (int i = 0; i < len1; ++ i) {
            l = i;
            r = i;
            while (l <= r && r <= s.length() - len1) {
                String str = idx2Str.get(r);
                if (marked.containsKey(str)) {
                    while (l <= r && marked.get(str) <= 0) {
                        String tmpStr = idx2Str.get(l);
                        if (marked.containsKey(tmpStr)) {
                            marked.put(tmpStr, marked.get(tmpStr) + 1);
                            ++ count;
                        }
                        l += len1;
                    }
                    -- count;
                    marked.put(str, marked.get(str) - 1);
                } else {
                    while (l <= r) {
                        String tmpStr = idx2Str.get(l);
                        if (marked.containsKey(tmpStr)) {
                            marked.put(tmpStr, marked.get(tmpStr) + 1);
                            ++ count;
                        }
                        l += len1;
                    }
                }
                if (count == 0) {
                    ans.add(l);
                    while (l <= r && count == 0) {
                        String tmpStr = idx2Str.get(l);
                        if (marked.containsKey(tmpStr)) {
                            marked.put(tmpStr, marked.get(tmpStr) + 1);
                            ++ count;
                        }
                        l += len1;
                    }
                }
                r += len1;
            }
            while (l <= r) {
                String tmpStr = idx2Str.get(l);
                if (marked.containsKey(tmpStr)) {
                    marked.put(tmpStr, marked.get(tmpStr) + 1);
                    ++ count;
                }
                l += len1;
            }
        }
        ans.sort(Integer::compare);
        return ans;
    }

    private HashMap<Integer, String> idx2Str = new HashMap<>();
}
```

#### [37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

题目：

编写一个程序，通过填充空格来解决数独问题。

数独的解法需 遵循如下规则：

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
数独部分空格内已填入了数字，空白格用 '.' 表示。



**示例：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/04/12/250px-sudoku-by-l2g-20050714svg.png)



解答：

很典型的回溯题，不同之处在于我们需要开辟行，列，块三个marked数组进行重复标记：

``` Java
class Solution {

    public void solveSudoku(char[][] board) {
        ans = board;
        for (int i = 0; i < board.length; ++i) {
            for (int j = 0; j < board[0].length; ++j) {
                if (board[i][j] != '.') {
                    int idx = board[i][j] - '0';
                    row[i][idx] = true;
                    column[j][idx] = true;
                    int m = i / 3, n = j / 3;
                    marked[m * 3 + n][idx] = true;
                }
            }
        }
        f(0, 0);
        board = ans;
    }

    private final boolean[][] row = new boolean[10][10];

    private final boolean[][] column = new boolean[10][10];

    private final boolean[][] marked = new boolean[10][10];

    private char[][] ans;

    private boolean f(int x, int y) {
        if (x == 9) {
            return true;
        }
        boolean flag = false;
        if (ans[x][y] != '.') {
            if (y == 8) {
                flag = f(x+1, 0);
            } else {
                flag = f(x, y+1);
            }
            return flag;
        }
        int idx = (x / 3) * 3 + y / 3;
        for (int i = 1; i <= 9; ++ i) {
            if (!row[x][i] && !column[y][i] && !marked[idx][i]) {
                row[x][i] = true;
                column[y][i] = true;
                marked[idx][i] = true;
                if (y == 8) {
                    flag = f(x+1, 0);
                } else {
                    flag = f(x, y+1);
                }
                if (flag) {
                    ans[x][y] = (char) (i + '0');
                    return true;
                }
                row[x][i] = false;
                column[y][i] = false;
                marked[idx][i] = false;
            }
        }
        return false;
    }
}
```

