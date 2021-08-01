package google50

import (
	"fmt"
	"math"
)

//DecodeString - P394 - TODO
// Ex1 3[a]2[b] -> aaabb
// Ex2 3[a2[c]] -> accaccacc
func DecodeString(in string) string {
	fmt.Println("Opening string ", in)
	if len(in) == 0 {
		return in
	}
	var i int
	var numStack, charStack, out []byte
	for i < len(in) {
		// if its a char, then add to acc
		if (in[i] >= 'a' && in[i] <= 'z') || in[i] == '[' {
			charStack = append(charStack, in[i])
			i++
		} else if in[i] > '0' && in[i] <= '9' {
			numStack = append(numStack, in[i])
			break
		} else {
			// matches ']' so we start popping off char stack
			for j := 0; j < len(charStack); j++ {

			}
		}
	}
	fmt.Println(string(out))
	return string(out)
}

// MaxScore - P 1423
func MaxScore(cardPoints []int, k int) int {
	best := sum(cardPoints[:k]...)
	current := best
	for i, j := k-1, len(cardPoints)-1; i >= 0; i, j = i-1, j-1 {
		current += cardPoints[j] - cardPoints[i]
		if current > best {
			best = current
		}
	}
	return best
}

// LongestOnes - calculates longest consecutive sequence of 1's with ability to flip "flips" number of 0's to 1
// We use the sliding window techniqiue, where right pointer first moves till we have encountered "flips" number of 0's
// then we move the left pointer
func LongestOnes(nums []int, flips int) int {
	ones, conversions := 0, 0
	for l, r := 0, 0; r < len(nums); r++ {
		if nums[r] == 0 {
			conversions++
		}
		for conversions > flips {
			if nums[l] == 0 {
				conversions--
			}
			l++
		}
		ones = max(ones, r-l+1)

	}
	return ones
}

// MaxPoints - P 149 (Unique points)
func MaxPoints(points [][]int) int {
	// calculate slope for all pairs of points and store the slope that in a hash map as a key
	// slope with most points will give us the count of max points on a line.
	// Note: we create a new hash map for each point in the first loop (this helps fix a point for each iteration and )
	var x1, y1, x2, y2, collinear int
	for i := 0; i < len(points); i++ {
		x1, y1 = points[i][0], points[i][1]
		for j := i + 1; j < len(points); j++ {
			x2, y2 = points[j][0], points[j][1]
			sy := y2 - y1
			sx := x2 - x1
			fmt.Println("Slope is ", sy, sx)
			// we cant store slope as float so we will instead store it as a string
			// we also need to make sure
		}
	}
	return collinear
}

func Gcd(a int, b int) int {
	var temp int
	for b > 0 {
		temp = b
		a, b = temp, a%b
	}
	return a
}

func MaxArea(grid [][]int) int {
	return 0
}

// EditDistance - calculates edit distance between 2 words
func editDistance(word1 string, word2 string) int {
	l1, l2 := len(word1), len(word2)
	edits := make([][]int, l1+1)
	// init edits array
	for i := 0; i < len(edits); i++ {
		edits[i] = make([]int, l2+1)
		for j := 0; j < len(edits[i]); j++ {
			if i == 0 {
				edits[i][j] = j
			}
			if j == 0 {
				edits[i][j] = i
			}
		}
	}
	// now iterate over the words and fill up edits array
	for i := 1; i < len(edits); i++ {
		for j := 1; j < len(edits[i]); j++ {
			if word1[i-1] == word2[j-1] {
				edits[i][j] = edits[i-1][j-1]
			} else {
				edits[i][j] = 1 + min(edits[i-1][j-1], edits[i-1][j], edits[i][j-1])
			}
		}
	}
	return edits[l1][l2]
}

type Pile struct {
	values   []int
	prevHead int
}

//PartitionSort - returns length of Longest Increasing Subsequence (using patience sort)
/*func PartitionSort(nums []int) int {
	piles := []int{}
	prev := math.MinInt32
	var numPiles int
	for _, n := range nums {
		if n > prev {
			pile = Pile{values: []int{n}, prevHead: prev}
			piles = append(piles, pile)
			numPiles++
		} else {
			pile = piles[numPiles-1]
			pile.values = append(pile.values, n)
		}
		prev = n
	}
	return numPiles
}*/

// LIS - return length of longest increasing subsequence using DP
func LIS(nums []int) int {
	lis := make([]int, len(nums))
	for i := 0; i < len(lis); i++ {
		lis[i] = 1
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				lis[i] = max(lis[i], 1+lis[j])
			}
		}
	}
	return max(lis...)
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
