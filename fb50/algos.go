package fb50

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

//P 636 - Exclusive time of functions
func exclusiveTime(n int, logs []string) []int {
	var prevId, currId, prevTime, currTime int
	var state string
	runTimes := make([]int, n)
	startingLog := strings.Split(logs[0], ":")
	prevId, _ = strconv.Atoi(startingLog[0])
	prevTime, _ = strconv.Atoi(startingLog[2])
	callStack := []int{prevId}
	var tail int
	for i := 1; i < len(logs); i++ {
		tail = len(callStack) - 1
		current := strings.Split(logs[i], ":")
		currId, _ = strconv.Atoi(current[0])
		state = current[1]
		currTime, _ = strconv.Atoi(current[2])
		if state == "start" {
			if len(callStack) > 0 {
				runTimes[callStack[tail]] += increment(currTime-1, prevTime)
			}
			callStack = append(callStack, currId)
			prevId, prevTime = currId, currTime
		} else {
			runTimes[callStack[tail]] += increment(currTime, prevTime)
			callStack = callStack[:tail]
			prevId, prevTime = currId, currTime+1
		}
	}
	return runTimes
}

func increment(end int, start int) int {
	if start == end {
		return 1
	}
	return end - start + 1
}

//P 66 - Plus one (add one to a number represented by an int array)
func plusOne(digits []int) []int {
	carry := 1
	for j := len(digits) - 1; j >= 0; j-- {
		if digits[j]+carry > 9 {
			digits[j] = 9 - digits[j]
			carry = 1
		} else {
			digits[j] += carry
			carry = 0
		}
	}
	if carry == 1 {
		digits = append([]int{1}, digits...)
	}
	return digits
}

func merge(intervals [][]int) [][]int {
	merged := [][]int{}
	// sort the intervals
	sort.SliceStable(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	start, end := intervals[0][0], intervals[0][1]
	for i := 1; i < len(intervals); i++ {
		if intervals[i][0] <= end {
			end = max(end, intervals[i][1])
		} else {
			merged = append(merged, []int{start, end})
			start, end = intervals[i][0], intervals[i][1]
		}
	}
	// merge last interval
	merged = append(merged, []int{start, end})
	return merged
}

//P42 - Trapping rain water
func trap(heights []int) int {
	leftHeights, rightHeights := make([]int, len(heights)), make([]int, len(heights))
	water := 0
	maxLeft, maxRight, minWallHeight := 0, 0, 0
	for i := range heights {
		maxLeft = max(maxLeft, heights[i])
		leftHeights[i] = maxLeft
	}
	j := len(heights) - 1
	for j >= 0 {
		maxRight = max(maxRight, heights[j])
		rightHeights[j] = maxRight
		j--
	}
	for i := range heights {
		minWallHeight = min(leftHeights[i], rightHeights[i])
		if minWallHeight > heights[i] {
			water += minWallHeight - heights[i]
		}
	}
	return water
}

//WordBreak - P139 (use DP)
func WordBreak(s string, wordDict []string) bool {
	wordMap := convertWordDictToMap(wordDict)
	var buffer strings.Builder
	matched := false
	left, w := 0, ""
	for i := 0; i < len(s); i++ {
		w = s[left : i+1]
		fmt.Println("Word is ", w)
		if _, ok := wordMap[w]; ok {
			fmt.Println("Matched")
			matched = true
		} else {
			if matched {
				buffer.WriteString(s[left:i])
				left = i
				matched = false
			}
		}
	}
	if matched {
		buffer.WriteString(w)
	}
	fmt.Println(s[left:])
	return buffer.String() == s
}

func convertWordDictToMap(words []string) map[string]bool {
	wordMap := make(map[string]bool)
	for _, w := range words {
		wordMap[w] = true
	}
	return wordMap
}

// FindKthSmallest - P215 using randomized select algo (can easily be converted to do kth largest)
func FindKthSmallest(nums []int, k int) int {
	pivot := nums[0]
	smaller, larger := []int{}, []int{}
	for _, n := range nums[1:] {
		if n <= pivot {
			smaller = append(smaller, n)
		} else {
			larger = append(larger, n)
		}
	}
	m := len(smaller)
	if k == m+1 {
		return pivot
	} else if k < m+1 {
		return FindKthSmallest(smaller, k)
	} else {
		return FindKthSmallest(larger, k-m)
	}
}

//P
func combinationSum(candidates []int, target int) [][]int {
	solutions := [][]int{}
	combinationsDfs(candidates, target, []int{}, &solutions)
	return unique(solutions)
}

func combinationsDfs(candidates []int, target int, state []int, solutions *[][]int) {
	if sum(state...) == target {
		*solutions = append(*solutions, state)
		return
	}
	if sum(state...) > target {
		return
	}

	for _, n := range candidates {
		tmp := make([]int, len(state), len(state)+1)
		copy(tmp, state)
		tmp = append(tmp, n)
		combinationsDfs(candidates, target, tmp, solutions)
		tmp = tmp[:len(tmp)-1]
	}
}

func unique(arrays [][]int) [][]int {
	res := [][]int{}
	arrMap := make(map[string][]int)
	for _, arr := range arrays {
		sort.Ints(arr)
		j, _ := json.Marshal(arr)
		str := string(j)
		if _, ok := arrMap[str]; !ok {
			arrMap[str] = arr
		}
	}
	for _, v := range arrMap {
		res = append(res, v)
	}
	return res
}

// P253 - Min. number of meeting rooms needed
func minMeetingRooms(intervals [][]int) int {
	starts, ends := make([]int, len(intervals)), make([]int, len(intervals))
	for i := 0; i < len(intervals); i++ {
		starts[i], ends[i] = intervals[i][0], intervals[i][1]
	}
	sort.Ints(starts)
	sort.Ints(ends)
	var i, j int
	var used int
	for i < len(starts) {
		if starts[i] >= ends[j] {
			j++
			used--
		}
		used++
		i++
	}
	return used
}

//P1249 - produce valid string after removing min. number of parentheses
func minRemoveToMakeValid(s string) string {
	stack := []int{}
	keepIdx := make([]bool, len(s))
	output := []byte{}
	for i := range s {
		if s[i] == '(' {
			stack = append(stack, i)
		} else if s[i] == ')' {
			if len(stack) > 0 {
				keepIdx[stack[len(stack)-1]] = true
				keepIdx[i] = true
				stack = stack[:len(stack)-1]
			}
		} else {
			keepIdx[i] = true
		}
	}
	for i := range s {
		if keepIdx[i] {
			output = append(output, s[i])
		}
	}
	return string(output)
}

// P977 - Squares of sorted arrays
func sortedSquares(nums []int) []int {
	inflectionIdx := 0
	squares := []int{}
	prev, square := 0, 0
	for i, n := range nums {
		square = n * n
		if square < prev {
			inflectionIdx = i
		}
		prev = square
		squares = append(squares, square)
	}
	// we split arrays at inflection point,
	arr1, arr2 := squares[:inflectionIdx], squares[inflectionIdx:]
	reverse(arr1)
	return mergeArr(arr1, arr2)
}

// merge 2 sorted arrays
func mergeArr(arr1 []int, arr2 []int) []int {
	res := []int{}
	i, j := 0, 0
	for i < len(arr1) && j < len(arr2) {
		if arr1[i] <= arr2[j] {
			res = append(res, arr1[i])
			i++
		} else {
			res = append(res, arr2[j])
			j++
		}
	}
	if i < len(arr1) {
		res = append(res, arr1[i:]...)
	}
	if j < len(arr2) {
		res = append(res, arr2[j:]...)
	}
	return res
}

//P360 - Sort transformed array (apply ax^2 + bx + c) to sort an already sorted array of integers
//TODO: what if cubic function
func sortTransformedArray(nums []int, a int, b int, c int) []int {
	left, right := 0, len(nums)-1
	var leftVal, rightVal int
	res := make([]int, len(nums))
	var idx int
	if a >= 0 {
		idx = right
	} else {
		idx = left
	}
	for left <= right {
		leftVal, rightVal = transform(nums[left], a, b, c), transform(nums[right], a, b, c)
		if a >= 0 {
			if leftVal > rightVal {
				res[idx] = leftVal
				left++
			} else {
				res[idx] = rightVal
				right--
			}
			idx--
		} else {
			if leftVal < rightVal {
				res[idx] = leftVal
				left++
			} else {
				res[idx] = rightVal
				right--
			}
			idx++
		}
	}
	return res
}

func transform(x int, a int, b int, c int) int {
	return x*x*a + x*b + c
}

func isAlienSorted(words []string, order string) bool {
	var k int
	for i := 1; i < len(words); i++ {
		w1, w2 := words[i-1], words[i]
		// iterate over both words to see which one is lexicographically smaller
		for k < len(w1) || k < len(w2) {
			if k == len(w1) {
				return true
			}
			if k == len(w2) {
				return false
			}
			p1, p2 := strings.IndexByte(order, w1[k]), strings.IndexByte(order, w2[k])
			if p1 > p2 {
				return false
			}
			if p1 < p2 {
				break
			}
			k++
		}
	}
	return true
}

//P1539 - Kth positive missing number
func findKthPositive(arr []int, k int) int {
	missingCount, prev := 0, 0
	for _, n := range arr {
		missingCount = n - prev - 1
		if missingCount >= k {
			return prev + k
		} else {
			k -= missingCount
		}
		prev = n
	}
	return arr[len(arr)-1] + k
}

// Generate Weighted Random Numbers
type Solution struct {
	Values      []int
	TotalWeight int
}

func Constructor(w []int) Solution {
	s := Solution{Values: []int{}, TotalWeight: 0}
	for _, weight := range w {
		s.Values = append(s.Values, weight)
		s.TotalWeight += weight
	}
	return s
}

func (s *Solution) PickIndex() int {
	r := rand.Intn(s.TotalWeight)
	sum := 0
	for i, n := range s.Values {
		sum += n
		if sum >= r {
			return i
		}
	}
	return len(s.Values) - 1
}

// P560 - count number of sub arrays whose sum equals k
func subarraySum(nums []int, k int) int {
	// calculate prefix sums
	hits := 0
	prefixSum, sum := make(map[int]int), 0
	for i := 0; i < len(nums); i++ {
		sum += nums[i]
		key := sum - k
		// if running sum is k, then increment count
		if key == 0 {
			hits++
		}
		if val, ok := prefixSum[key]; ok {
			hits += val
		}
		// increment the number of sub arrays whose prefix sum is sum
		if _, ok := prefixSum[sum]; ok {
			prefixSum[sum]++
		} else {
			prefixSum[sum] = 1
		}
	}
	return hits
}

//P139 - WordBreak (TODO: improve solution)
func wordBreak(s string, wordList []string) bool {
	memo := map[int]bool{}
	return dfsWordBreak(s, 0, wordList, memo)

}

func dfsWordBreak(s string, curr int, wordList []string, memo map[int]bool) bool {
	if curr == len(s) {
		return true
	}
	for _, w := range wordList {
		if strings.HasPrefix(s[curr:], w) {
			if _, ok := memo[curr+len(w)]; !ok {
				memo[curr+len(w)] = dfsWordBreak(s, curr+len(w), wordList, memo)
				if memo[curr+len(w)] {
					return true
				}
			}
		}
	}
	return false
}

func ProductExceptSelf(nums []int) []int {
	forward, backward, results := make([]int, len(nums)), make([]int, len(nums)), make([]int, len(nums))
	forward[0] = 1
	// forward traversal
	for i := 1; i < len(nums); i++ {
		forward[i] = forward[i-1] * nums[i-1]
	}
	// backward traversal
	backward[len(nums)-1] = 1
	for i := len(nums) - 2; i >= 0; i-- {
		backward[i] = backward[i+1] * nums[i+1]
	}
	for i := 0; i < len(nums); i++ {
		results[i] = forward[i] * backward[i]
	}
	return results
}

// MARK: BST problems

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type TreeLevel struct {
	Node  *TreeNode
	Level int
}

// RHS - Givs right hand side view of a binary tree (BT)
func RHS(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	// need to perform level order traversal and for each level append max value
	treeLevel := TreeLevel{Node: root, Level: 0}
	queue := []TreeLevel{treeLevel}
	var levelValues []int
	var rhs []int
	for len(queue) > 0 {
		// pop node off the queue
		treeLevel, queue = queue[0], queue[1:]
		// add children to queue for BFS
		if treeLevel.Node.Left != nil {
			queue = append(queue, TreeLevel{Node: treeLevel.Node.Left, Level: treeLevel.Level + 1})
		}
		if treeLevel.Node.Right != nil {
			queue = append(queue, TreeLevel{Node: treeLevel.Node.Right, Level: treeLevel.Level + 1})
		}
		levelValues = append(levelValues, treeLevel.Node.Val)
		// check if we are at the end of a level
		if len(queue) > 0 && treeLevel.Level < queue[0].Level {
			rhs = append(rhs, levelValues[len(levelValues)-1])
			// reset level values
			levelValues = []int{}
		}
	}
	if len(levelValues) > 0 {
		rhs = append(rhs, levelValues[len(levelValues)-1])
	}
	return rhs
}

// P103 - ZigZag Traversal (same as level order just with odd numbered level values reversed)
func zigzagLevelOrder(root *TreeNode) [][]int {
	results := [][]int{}
	if root == nil {
		return results
	}
	treeLevel := TreeLevel{Level: 0, Node: root}
	currLevel, currValues := 0, []int{}
	Q := []TreeLevel{TreeLevel{Level: currLevel, Node: root}}
	for len(Q) > 0 {
		treeLevel, Q = Q[0], Q[1:]
		if treeLevel.Level != currLevel {
			if currLevel%2 != 0 {
				reverse(currValues)
			}
			results = append(results, currValues)
			currValues = []int{}
		}
		currLevel = treeLevel.Level
		currValues = append(currValues, treeLevel.Node.Val)
		if treeLevel.Node.Left != nil {
			Q = append(Q, TreeLevel{Level: treeLevel.Level + 1, Node: treeLevel.Node.Left})
		}
		if treeLevel.Node.Right != nil {
			Q = append(Q, TreeLevel{Level: treeLevel.Level + 1, Node: treeLevel.Node.Right})
		}
	}
	// clean up final values left
	if currLevel%2 != 0 {
		reverse(currValues)
	}
	results = append(results, currValues)
	return results
}

// RangeSumBST - P938 (go over a given range and calculate sum over that range)
// TIP: for recursions where you feel the need for an accumulator, try only doing comparisons at teh root node
func RangeSumBST(root *TreeNode, low int, high int) int {
	// do inorder traversal while discarding nodes outside the range
	sum, leftVal, rightVal := 0, 0, 0
	if root.Left != nil {
		leftVal = RangeSumBST(root.Left, low, high)
	}
	if root.Val >= low && root.Val <= high {
		sum += root.Val
	}
	if root.Right != nil {
		rightVal = RangeSumBST(root.Right, low, high)
	}
	return sum + leftVal + rightVal
}

// P449 - Serialize/Deserialize BST

//serialize tree with pre order traversal
func preorder(root *TreeNode) string {
	if root == nil {
		return ""
	}
	var sb strings.Builder
	val := strconv.Itoa(root.Val)
	sb.WriteString(val)
	sb.WriteString(",")
	if root.Left != nil {
		leftVal := preorder(root.Left)
		sb.WriteString(leftVal)

	}
	if root.Right != nil {
		rightVal := preorder(root.Right)
		sb.WriteString(rightVal)
	}
	return sb.String()
}

// Deserializes your encoded data to tree.
func deserialize(data string) *TreeNode {
	preorder := []int{}
	for _, s := range strings.Split(data, ",") {
		if s == "" {
			continue
		}
		n, _ := strconv.Atoi(s)
		preorder = append(preorder, n)
	}
	tree, _ := constructTree(preorder, 0, math.MinInt64, math.MaxInt64)
	return tree
}

func constructTree(preorder []int, currIdx int, min int, max int) (*TreeNode, int) {
	if currIdx > len(preorder)-1 {
		return nil, currIdx
	}
	val := preorder[currIdx]
	if val < min || val > max {
		return nil, currIdx
	}

	root := &TreeNode{Val: val}
	currIdx++
	root.Left, currIdx = constructTree(preorder, currIdx, min, root.Val)
	root.Right, currIdx = constructTree(preorder, currIdx, root.Val, max)
	return root, currIdx
}

func inorder(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	traversal := []int{}
	if root.Left != nil {
		leftTraversal := inorder(root.Left)
		traversal = append(traversal, leftTraversal...)
	}
	traversal = append(traversal, root.Val)
	if root.Right != nil {
		rightTraversal := inorder(root.Right)
		traversal = append(traversal, rightTraversal...)
	}
	return traversal
}

func postorder(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	traversal := []int{}
	if root.Left != nil {
		leftTraversal := postorder(root.Left)
		traversal = append(traversal, leftTraversal...)
	}
	if root.Right != nil {
		rightTraversal := postorder(root.Right)
		traversal = append(traversal, rightTraversal...)
	}
	traversal = append(traversal, root.Val)
	return traversal
}

// P 1448 - count the number of good nodes
func goodNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return countGoodNodes(root, math.MinInt32)
}

func countGoodNodes(root *TreeNode, maxVal int) int {
	if root == nil {
		return 0
	}
	var count int
	if root.Val >= maxVal {
		count++
	}
	if root.Left != nil {
		count += countGoodNodes(root.Left, max(root.Val, maxVal))
	}
	if root.Right != nil {
		count += countGoodNodes(root.Right, max(root.Val, maxVal))
	}
	return count
}

// P543 - Diameter of Binary tree (max distance between any 2 nodes)
func diameterOfBinaryTree(root *TreeNode) int {
	_, maxDiameter := diameterHelper(root)
	return maxDiameter - 1
}

func diameterHelper(root *TreeNode) (int, int) {
	if root == nil {
		return 0, 0
	}
	var leftDiameter, maxLeft, rightDiameter, maxRight int
	if root.Left != nil {
		leftDiameter, maxLeft = diameterHelper(root.Left)
	}
	if root.Right != nil {
		rightDiameter, maxRight = diameterHelper(root.Right)
	}
	rootDiameter := max(leftDiameter, rightDiameter) + 1
	maxDiameter := max(leftDiameter+rightDiameter+1, maxLeft, maxRight)
	return rootDiameter, maxDiameter
}

// P98 - Validate BST
func isValidBST(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return validate(root, math.MinInt64, math.MaxInt64)
}

func validate(root *TreeNode, min int, max int) bool {
	leftValid, rightValid := true, true
	if root.Left != nil {
		leftValid = validate(root.Left, min, root.Val)
	}
	if root.Right != nil {
		rightValid = validate(root.Right, root.Val, max)
	}
	return root.Val > min && root.Val < max && leftValid && rightValid

}

//P124 - Max path sum of a BT (similar as diameter but max sum requires more careful checking due to negative numbers)
func maxPathSum(root *TreeNode) int {
	_, maxSum := pathSumHelper(root)
	return maxSum
}

func pathSumHelper(root *TreeNode) (int, int) {
	if root == nil {
		return 0, math.MinInt32
	}
	var leftPathSum, rightPathSum int
	maxLeft, maxRight := math.MinInt32, math.MinInt32
	if root.Left != nil {
		leftPathSum, maxLeft = pathSumHelper(root.Left)
	}
	if root.Right != nil {
		rightPathSum, maxRight = pathSumHelper(root.Right)
	}
	pathSum := max(leftPathSum, rightPathSum, 0) + root.Val
	maxSum := max(leftPathSum+rightPathSum+root.Val, leftPathSum+root.Val, rightPathSum+root.Val, root.Val, maxLeft, maxRight)
	return pathSum, maxSum
}

// MARK: Graphs

// GraphI represents graph with integer vertices
type GraphI map[int][]int

// BFS - Breadth first traversal of a graph
func BFS(G GraphI, source int) map[int]int {
	var node int
	seen, pred := make(map[int]bool), make(map[int]int)
	Q := []int{source}
	for len(Q) > 0 {
		node, Q = Q[0], Q[1:]
		neighbors, _ := G[node]
		for _, neighbor := range neighbors {
			if seen[neighbor] {
				continue
			}
			seen[neighbor] = true
			Q = append(Q, neighbor)
			pred[neighbor] = node
		}
	}
	return pred
}

// P207 - Course Schedule (Top sort with cycle detection) - solution uses Kahn's topsort
func canFinish(numCourses int, prereqs [][]int) bool {
	courses := constructCourseGraph(numCourses, prereqs)
	sorted := []int{}
	inDegrees := make(map[int]int)
	for p, edges := range courses {
		if _, ok := inDegrees[p]; !ok {
			inDegrees[p] = 0
		}
		for _, v := range edges {
			if _, ok := inDegrees[v]; !ok {
				inDegrees[v] = 0
			}
			inDegrees[v]++
		}
	}
	Q := []int{}
	// identify vertices with zero indegree
	for v, degree := range inDegrees {
		if degree == 0 {
			Q = append(Q, v)
			sorted = append(sorted, v)
			delete(inDegrees, v)
		}
	}

	// now traverse the graph starting from these 0 indegree vertices
	var vertice int
	for len(Q) > 0 {
		vertice, Q = Q[0], Q[1:]
		edges, _ := courses[vertice]
		for _, v := range edges {
			inDegrees[v] -= 1
			if inDegrees[v] == 0 {
				Q = append(Q, v)
				sorted = append(sorted, v)
				delete(inDegrees, v)
			}
		}
	}
	return len(sorted) == numCourses
}

func constructCourseGraph(numCourses int, prereqs [][]int) map[int][]int {
	G := make(map[int][]int)
	// init graph edges
	for i := 0; i < numCourses; i++ {
		G[i] = []int{}
	}
	var i, j int
	for _, vertices := range prereqs {
		i, j = vertices[0], vertices[1]
		G[j] = append(G[j], i)
	}
	return G
}

// MARK: Backtracking

// P46 - generate permutations
func permute(nums []int) [][]int {
	used := make([]bool, len(nums))
	res := [][]int{}
	dfsPermutations(nums, used, []int{}, &res)
	return res
}

func dfsPermutations(nums []int, used []bool, state []int, res *[][]int) {
	if len(state) == len(nums) {
		*res = append(*res, state)
		return
	}
	for i, n := range nums {
		if used[i] {
			continue
		}
		tmp := make([]int, len(state), len(state)+1)
		copy(tmp, state)
		tmp = append(tmp, n)
		used[i] = true
		dfsPermutations(nums, used, tmp, res)
		used[i] = false
	}
}

//P 67 - Add Binary
func addBinary(a string, b string) string {
	if a == "0" && b == "0" {
		return "0"
	}
	var carry int
	i, j := len(a)-1, len(b)-1
	var res strings.Builder
	for i >= 0 || j >= 0 {
		if i >= 0 && a[i] == '1' {
			carry++
		}
		if j >= 0 && b[j] == '1' {
			carry++
		}
		if carry%2 == 1 {
			res.WriteString("1")
		} else {
			res.WriteString("0")
		}
		carry = carry / 2
		i--
		j--
	}
	if carry == 1 {
		return "1" + reverse(res.String())
	}
	return reverseStr(res.String())
}

func reverseStr(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}

// MARK: Utils

func binarySearch(nums []int, key int) int {
	low, high := 0, len(nums)-1
	for low <= high {
		mid := (low + high) / 2
		if nums[mid] == key {
			return mid
		} else if nums[mid] > key {
			high = mid - 1
		} else {
			low = mid + 1
		}
	}
	return -1
}

func sum(nums ...int) int {
	var sum int
	for _, n := range nums {
		sum += n
	}
	return sum
}

func max(nums ...int) int {
	var maximum = math.MinInt32
	for _, n := range nums {
		if maximum < n {
			maximum = n
		}
	}
	return maximum
}

func min(nums ...int) int {
	minimum := math.MaxInt32
	for _, n := range nums {
		if minimum > n {
			minimum = n
		}
	}
	return minimum
}

func reverse(nums []int) {
	for i, j := 0, len(nums)-1; i < j; i, j = i+1, j-1 {
		nums[i], nums[j] = nums[j], nums[i]
	}
}
