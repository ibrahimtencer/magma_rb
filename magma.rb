#!/usr/bin/env ruby
require 'prime'

def linear a, b, modulus
  -> x, y {(a*x + b*y) % modulus}
end

def linear_array a, b, modulus
  (0...modulus).map {|x| (0...modulus).map {|y| (a*x + b*y) % modulus}}
end

def inverse x, modulus
  (1...modulus).each do |i|
    return i if (i * x) % modulus == 1
  end
end

Expx = -> f, x, y {x}

Equations =
  {1518 => [Expx, -> f, x, y {f[f[y, y], f[x, f[y, x]]]}],
   #x = x(x(xx)) = x(xx²)
   47 => [Expx, -> f, x, y {f[x, f[x, f[x, x]]]}],
   #x = x(x((xx)x))
   614 => [Expx, -> f, x, y {f[x, f[x, f[f[x, x], x]]]}],
   #x = x((xx)(xx))
   817 => [Expx, -> f, x, y {f[x, f[f[x, x], f[x, x]]]}],
   #xx = (x(xx))x
   3862 => [-> f, x, y {f[x, x]}, -> f, x, y {f[f[x, f[x, x]], x]}],
   677 => [Expx, -> f, x, y {f[y, f[x, f[f[y, x], y]]]}],
   255 => [Expx, -> f, x, y {f[f[f[x, x], x], x]}]
  }

def b_candidate? eqn, b, modulus
  case eqn
  when 1518
    #(1 - b³)² + b² - b⁵ + b³(1 - b³) = 0
    ((1 - b*b*b)**2 + b**2 - b**5 + b**3 * (1 - b**3)) % modulus == 0
  when 677
    #0 = 1 + b + b² + b⁴ + 2b⁶ + b⁸
    (1 + b + b*b + b**4 + 2*b**6 + b**8) % modulus == 0
  end
end

def b_candidates eqn, modulus
  (0...modulus).select {|n| b_candidate?(eqn, n, modulus)}
end

def check op, eqn, lhs_fn, rhs_fn, x, y
  lhs = lhs_fn[op, x, y]
  rhs = rhs_fn[op, x, y]
  if lhs != rhs
    p [x, y, lhs, rhs, "#{eqn} failed"]
  end
end

def calc_a eqn, b, m
  #m is the modulus
  case eqn
  when 1518
    b_inv = inverse(b, m) #(m + 1)/b
    ((1 - b*b*b) * b_inv) % m
  when 677
    #a = 1/b(1 + b²)
    (inverse(b, m) * inverse(1 + b*b, m)) % m
  end
end

$none = []

def search_linear antecedent = 677, cons = [255], upto: 20, describe: false, describe_op: false
  Prime.each do |m|
    return if m > upto
    bcand = b_candidates(antecedent, m)
    if bcand.empty?
      puts "no candidates for #{m}"
      $none << m
    end
    bcand.each do |b|
      a = calc_a(antecedent, b, m)
      op = linear(a, b, m)
      puts "modulus: #{m}, a: #{a}, b: #{b}"

      (1...m).each do |x|
        (1...m).each do |y|
          ([antecedent] + cons).each do |eqn|
            check(op, eqn.to_s, Equations[eqn][0], Equations[eqn][1], x, y)
          end
        end
      end

      describe(linear_array(a, b, m)) if describe
      describe(transpose(linear_array(a, b, m))) if describe_op
    end
  end
end

def completions table
end

def find_such_that modulus, &blk
  blk.call
end

def sub_search677poly m
  #c(ab²c + b² + b + 1) = 0
  test1 = -> ac, b {ac*b**2 + b**2 + b + 1 == 0}
  set = 0...m
  cands = []
  set.each {|ac| set.each {|b| cands << [ac, b] if test1[ac, b]}}
  cands
end

def search677poly m
  res = []
  (2...m).each do |i|
    cands = sub_search677poly(i)
    if cands.empty?
      puts "none for #{i}"
    else
      p cands
      res << cands
    end
  end
  res
end

#x = (yy)(x(yx))
def find_1518 n
  #a finite 1518 magma must have each left-multiplication a bijection so we take some permutation as L_x.
  #squaring must also be a bijection
  op_tables = []
  ks = 0...n
  table = ks.map {[nil] * n}
  ks.permutations do |perm|
  end
end

def table_from_values vals
  n = vals.flatten(2).max #can use sqrt size too
  p vals
  (0..n).map {|x| (0..n).map {|y| vals[[x, y]]}}
end

def fmb_to_array(str)
  #take output from Vampire fmb mode and make it a multiplication table
  #Vampire makes it 1-indexed, we make 0-indexed
  #exclude nil values - some lines may not have an assignment on them
  vals = str.split("\n").map {|line| line.scan(/\d+/).map(&:to_i).map {|i| i - 1}}.map {|i, j, k| [[i, j], k]}.select {|x, y| y}.to_h
  #n = entries.flatten(2).max
  #table = (0..n).map {|x| (0..n).map {|y| entries[[x, y]]}}
  table_from_values(vals)
end

def line_break_array(ar)
  "[#{ar.map(&:to_s).join(",\n")}]"
end

def nice_fmb_to_array str
  ar = fmb_to_array(str)
  puts line_break_array(ar)
  ar
end

def cycles perm, show_fixpoints=false
  #gives the cycles of a permutation (array of numbers), as an array of arrays
  res = []
  current = [0]
  while 1
    nxt = perm[current[-1]]
    if nxt == current[0]
      res << current #if current.size > 1 || show_fixpoints
      k = (perm - res.flatten).min
      if !k
        return show_fixpoints ? res : res.select {|cyc| cyc.size > 1}
      end
      current = [k]
    else
      current << nxt
    end
  end
end

def pp_cycles cycles, sep = " "
  #pretty print
  cycles.map {|cycle| "(#{cycle.map(&:to_s).join(sep)})"}.join
end

def invert_group hash
  #returns a hash whose keys are hash's values, with the array of keys that went to it as the values
  res = {}
  hash.values.each do |v|
    res[v] = hash.keys.select {|k| hash[k] == v}
  end
  res
end

def p_as_set array
  #[] => {}
  "{#{array.join(", ")}}"
end

def gen_cyclic_tptp n
  domain = (0...n).to_a
  axs = ["fof(dom, axiom,\n  #{domain.map {|i| "X = #{i}"}.join(" | ")}\n).",
         "fof(distinct, axiom,\n  #{domain.map {|i| ((i+1)...n).map {|j| "#{i} != #{j}"}}.flatten.join(" & ")}\n).",
         "fof(succ, axiom,\n  #{domain.map {|i| "s(#{i}) = #{i == n - 1 ? 0 : i + 1}"}.join(" & ")}\n).",
         "fof(def_S, axiom,\n  s(X) = m(X, X)\n).",
         "fof(eq1518, axiom,\n  X = m(s(Y), m(X, m(Y, X)))\n).",
         "fof(left_mul_inj, axiom,\n  m(Z, Y) = m(Z, W) => Y = W\n).", #speeds it up hugely
         #"fof(left_mul_surj, axiom, ![Z] : ?[Y] : m(Z, Y) = X).", #doesn't speed up
         "fof(squaring_hom, axiom,\n  ![X, Y] : m(s(X), s(Y)) = s(m(X, Y))\n).", #also huge speedup
         #xSx = S¯¹x
         "fof(xsx_eq_s_inv_x, axiom, s(m(x, s(x))) = x).",

         #"fof(s3, conjecture,\n  0 = 3\n).",
  ]
  axs.join("\n\n")
end

def table_to_values table
  domain = interval(table.size)
  domain.product(domain).map {|i, j| [[i, j], table[i][j]]}.to_h
end

def interval n
  (0...n).to_a
end

def renumber table, ar
  #renumber the elements according to array of numbers ar
  #could also use the inverse permutation (ar[x], ar[y])
  vals = table_to_values(table)
  vals = vals.map {|(x, y), out| [[ar.index(x), ar.index(y)], ar.index(out)]}.to_h
  table_from_values(vals)
end

def transpose table
  domain = interval(table.size)
  domain.map {|i| table.map {|row| row[i]}}
end

def injective? seq
  seq.uniq == seq
end

def describe table, show_fixpoints=false
  #describe the multiplication table for a 1518 magma
  #assumes that it's given as an array of arrays of numbers in [0, n)
  #describe(fmb_to_array("..."))
  n = table.size
  #puts line_break_array(table)
  sep = n > 10 ? " " : ""
  puts "size = #{n}"
  set = interval(n)
  square = set.map {|i| table[i][i]}
  square_cycles = cycles(square, show_fixpoints) if injective?(square)
  left_is_square = set.select {|y| table.all? {|row| row[y] == square[y]}}
  other_lefts = invert_group((set).map {|x| [x, table[x]]}.to_h)
  has_lis = !left_is_square.empty?
  sq_frac = set.sum {|x| set.count {|y| table[x][y] == square[y]}}
  puts "square frac: #{sq_frac}/#{n**2} = #{sq_frac.to_f / n**2}"
  puts("A = #{p_as_set(left_is_square)}") if has_lis
  puts("Sx = #{pp_cycles(square_cycles, sep)}") if square_cycles
  puts "x*y ="
  puts("      Sy if y ∈ A") if has_lis
  other_lefts.each do |left, xs|
    puts "      #{pp_cycles(cycles(left, show_fixpoints) - (has_lis ? [] || square_cycles : []), sep)}y if x #{xs.size > 1 ? "∈ #{xs == left_is_square ? "A" : p_as_set(xs)}" : "= #{xs[0]}"}"
  end
  nil
end

Shom8 = [[2, 3, 4, 7, 0, 1, 5, 6],
[2, 1, 4, 5, 0, 3, 6, 7],
[2, 3, 4, 7, 0, 1, 5, 6],
[2, 7, 4, 3, 0, 5, 6, 1],
[2, 3, 4, 7, 0, 1, 5, 6],
[2, 6, 4, 3, 0, 5, 1, 7],
[2, 1, 4, 3, 0, 7, 6, 5],
[2, 1, 4, 6, 0, 5, 3, 7]]

Shom9 = [[4, 1, 3, 7, 5, 0, 8, 2, 6],
[3, 1, 4, 5, 7, 2, 6, 0, 8],
[4, 6, 3, 7, 5, 0, 1, 2, 8],
[4, 6, 3, 7, 5, 0, 1, 2, 8],
[4, 1, 3, 7, 5, 0, 8, 2, 6],
[4, 1, 3, 7, 5, 0, 8, 2, 6],
[2, 1, 5, 0, 3, 7, 6, 4, 8],
[4, 6, 3, 7, 5, 0, 1, 2, 8],
[7, 1, 0, 4, 2, 3, 6, 5, 8]]

Shom10 = [[0, 2, 8, 6, 9, 5, 1, 3, 7, 4],
[4, 3, 6, 8, 5, 9, 7, 2, 1, 0],
[5, 3, 6, 8, 4, 0, 7, 2, 1, 9],
[4, 3, 6, 8, 5, 9, 7, 2, 1, 0],
[5, 3, 6, 8, 4, 0, 7, 2, 1, 9],
[0, 3, 6, 8, 9, 5, 7, 2, 1, 4],
[5, 3, 6, 8, 4, 0, 7, 2, 1, 9],
[5, 3, 6, 8, 4, 0, 7, 2, 1, 9],
[4, 3, 6, 8, 5, 9, 7, 2, 1, 0],
[5, 7, 1, 2, 4, 0, 3, 8, 6, 9]]

Shom11 = [[0, 1, 4, 8, 5, 2, 9, 7, 6, 3, 10],
[6, 1, 4, 3, 5, 2, 8, 7, 9, 0, 10],
[0, 3, 4, 1, 5, 2, 6, 10, 8, 9, 7],
[6, 1, 4, 3, 5, 2, 8, 10, 9, 0, 7],
[0, 3, 4, 1, 5, 2, 6, 10, 8, 9, 7],
[0, 3, 4, 1, 5, 2, 6, 10, 8, 9, 7],
[3, 7, 4, 9, 5, 2, 6, 1, 0, 8, 10],
[9, 1, 4, 0, 5, 2, 3, 7, 8, 6, 10],
[9, 1, 4, 0, 5, 2, 3, 10, 8, 6, 7],
[8, 10, 4, 6, 5, 2, 0, 7, 3, 9, 1],
[8, 1, 4, 6, 5, 2, 0, 7, 3, 9, 10]]

Shom12 = [[6, 4, 11, 0, 8, 10, 3, 2, 7, 5, 9, 1],
[9, 2, 8, 10, 11, 0, 5, 4, 1, 3, 6, 7],
[9, 2, 8, 10, 11, 0, 5, 4, 1, 3, 6, 7],
[6, 4, 11, 0, 8, 10, 3, 2, 7, 5, 9, 1],
[10, 2, 8, 5, 11, 6, 9, 4, 1, 0, 3, 7],
[6, 11, 7, 0, 2, 10, 3, 1, 4, 5, 9, 8],
[6, 4, 11, 0, 8, 10, 3, 2, 7, 5, 9, 1],
[10, 2, 8, 5, 11, 6, 9, 4, 1, 0, 3, 7],
[9, 2, 8, 10, 11, 0, 5, 4, 1, 3, 6, 7],
[6, 11, 7, 0, 2, 10, 3, 1, 4, 5, 9, 8],
[6, 11, 7, 0, 2, 10, 3, 1, 4, 5, 9, 8],
[10, 2, 8, 5, 11, 6, 9, 4, 1, 0, 3, 7]]

Shom13 = [[0, 5, 4, 8, 10, 11, 3, 2, 9, 1, 6, 12, 7],
[0, 6, 12, 10, 11, 9, 7, 1, 4, 2, 5, 8, 3],
[0, 4, 5, 2, 7, 3, 11, 8, 6, 10, 12, 1, 9],
[0, 4, 5, 2, 7, 3, 11, 8, 6, 10, 12, 1, 9],
[0, 6, 9, 12, 11, 10, 7, 1, 4, 3, 2, 8, 5],
[0, 4, 5, 2, 7, 3, 11, 8, 6, 10, 12, 1, 9],
[0, 6, 12, 10, 11, 9, 7, 1, 4, 2, 5, 8, 3],
[0, 6, 12, 10, 11, 9, 7, 1, 4, 2, 5, 8, 3],
[0, 6, 9, 12, 11, 10, 7, 1, 4, 3, 2, 8, 5],
[0, 8, 5, 2, 1, 3, 4, 11, 7, 10, 12, 6, 9],
[0, 8, 5, 2, 1, 3, 4, 11, 7, 10, 12, 6, 9],
[0, 6, 9, 12, 11, 10, 7, 1, 4, 3, 2, 8, 5],
[0, 8, 5, 2, 1, 3, 4, 11, 7, 10, 12, 6, 9]]

Ord25_677 = [[1, 2, 3, 4, 5, 6, 0, 28, 29, 30, 31, 32, 33, 34, 21, 22, 23, 24, 25, 26, 27, 14, 15, 16, 17, 18, 19, 20, 7, 8, 9, 10, 11, 12, 13],
[5, 6, 0, 1, 2, 3, 4, 32, 33, 34, 28, 29, 30, 31, 25, 26, 27, 21, 22, 23, 24, 18, 19, 20, 14, 15, 16, 17, 11, 12, 13, 7, 8, 9, 10],
[2, 3, 4, 5, 6, 0, 1, 29, 30, 31, 32, 33, 34, 28, 22, 23, 24, 25, 26, 27, 21, 15, 16, 17, 18, 19, 20, 14, 8, 9, 10, 11, 12, 13, 7],
[6, 0, 1, 2, 3, 4, 5, 33, 34, 28, 29, 30, 31, 32, 26, 27, 21, 22, 23, 24, 25, 19, 20, 14, 15, 16, 17, 18, 12, 13, 7, 8, 9, 10, 11],
[3, 4, 5, 6, 0, 1, 2, 30, 31, 32, 33, 34, 28, 29, 23, 24, 25, 26, 27, 21, 22, 16, 17, 18, 19, 20, 14, 15, 9, 10, 11, 12, 13, 7, 8],
[0, 1, 2, 3, 4, 5, 6, 34, 28, 29, 30, 31, 32, 33, 27, 21, 22, 23, 24, 25, 26, 20, 14, 15, 16, 17, 18, 19, 13, 7, 8, 9, 10, 11, 12],
[4, 5, 6, 0, 1, 2, 3, 31, 32, 33, 34, 28, 29, 30, 24, 25, 26, 27, 21, 22, 23, 17, 18, 19, 20, 14, 15, 16, 10, 11, 12, 13, 7, 8, 9],
[14, 15, 16, 17, 18, 19, 20, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 28, 29, 30, 31, 32, 33, 34, 21, 22, 23, 24, 25, 26, 27],
[18, 19, 20, 14, 15, 16, 17, 11, 12, 13, 7, 8, 9, 10, 4, 5, 6, 0, 1, 2, 3, 32, 33, 34, 28, 29, 30, 31, 25, 26, 27, 21, 22, 23, 24],
[15, 16, 17, 18, 19, 20, 14, 8, 9, 10, 11, 12, 13, 7, 1, 2, 3, 4, 5, 6, 0, 29, 30, 31, 32, 33, 34, 28, 22, 23, 24, 25, 26, 27, 21],
[19, 20, 14, 15, 16, 17, 18, 12, 13, 7, 8, 9, 10, 11, 5, 6, 0, 1, 2, 3, 4, 33, 34, 28, 29, 30, 31, 32, 26, 27, 21, 22, 23, 24, 25],
[16, 17, 18, 19, 20, 14, 15, 9, 10, 11, 12, 13, 7, 8, 2, 3, 4, 5, 6, 0, 1, 30, 31, 32, 33, 34, 28, 29, 23, 24, 25, 26, 27, 21, 22],
[20, 14, 15, 16, 17, 18, 19, 13, 7, 8, 9, 10, 11, 12, 6, 0, 1, 2, 3, 4, 5, 34, 28, 29, 30, 31, 32, 33, 27, 21, 22, 23, 24, 25, 26],
[17, 18, 19, 20, 14, 15, 16, 10, 11, 12, 13, 7, 8, 9, 3, 4, 5, 6, 0, 1, 2, 31, 32, 33, 34, 28, 29, 30, 24, 25, 26, 27, 21, 22, 23],
[28, 29, 30, 31, 32, 33, 34, 21, 22, 23, 24, 25, 26, 27, 14, 15, 16, 17, 18, 19, 20, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6],
[32, 33, 34, 28, 29, 30, 31, 25, 26, 27, 21, 22, 23, 24, 18, 19, 20, 14, 15, 16, 17, 11, 12, 13, 7, 8, 9, 10, 4, 5, 6, 0, 1, 2, 3],
[29, 30, 31, 32, 33, 34, 28, 22, 23, 24, 25, 26, 27, 21, 15, 16, 17, 18, 19, 20, 14, 8, 9, 10, 11, 12, 13, 7, 1, 2, 3, 4, 5, 6, 0],
[33, 34, 28, 29, 30, 31, 32, 26, 27, 21, 22, 23, 24, 25, 19, 20, 14, 15, 16, 17, 18, 12, 13, 7, 8, 9, 10, 11, 5, 6, 0, 1, 2, 3, 4],
[30, 31, 32, 33, 34, 28, 29, 23, 24, 25, 26, 27, 21, 22, 16, 17, 18, 19, 20, 14, 15, 9, 10, 11, 12, 13, 7, 8, 2, 3, 4, 5, 6, 0, 1],
[34, 28, 29, 30, 31, 32, 33, 27, 21, 22, 23, 24, 25, 26, 20, 14, 15, 16, 17, 18, 19, 13, 7, 8, 9, 10, 11, 12, 6, 0, 1, 2, 3, 4, 5],
[31, 32, 33, 34, 28, 29, 30, 24, 25, 26, 27, 21, 22, 23, 17, 18, 19, 20, 14, 15, 16, 10, 11, 12, 13, 7, 8, 9, 3, 4, 5, 6, 0, 1, 2],
[7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 28, 29, 30, 31, 32, 33, 34, 21, 22, 23, 24, 25, 26, 27, 14, 15, 16, 17, 18, 19, 20],
[11, 12, 13, 7, 8, 9, 10, 4, 5, 6, 0, 1, 2, 3, 32, 33, 34, 28, 29, 30, 31, 25, 26, 27, 21, 22, 23, 24, 18, 19, 20, 14, 15, 16, 17],
[8, 9, 10, 11, 12, 13, 7, 1, 2, 3, 4, 5, 6, 0, 29, 30, 31, 32, 33, 34, 28, 22, 23, 24, 25, 26, 27, 21, 15, 16, 17, 18, 19, 20, 14],
[12, 13, 7, 8, 9, 10, 11, 5, 6, 0, 1, 2, 3, 4, 33, 34, 28, 29, 30, 31, 32, 26, 27, 21, 22, 23, 24, 25, 19, 20, 14, 15, 16, 17, 18],
[9, 10, 11, 12, 13, 7, 8, 2, 3, 4, 5, 6, 0, 1, 30, 31, 32, 33, 34, 28, 29, 23, 24, 25, 26, 27, 21, 22, 16, 17, 18, 19, 20, 14, 15],
[13, 7, 8, 9, 10, 11, 12, 6, 0, 1, 2, 3, 4, 5, 34, 28, 29, 30, 31, 32, 33, 27, 21, 22, 23, 24, 25, 26, 20, 14, 15, 16, 17, 18, 19],
[10, 11, 12, 13, 7, 8, 9, 3, 4, 5, 6, 0, 1, 2, 31, 32, 33, 34, 28, 29, 30, 24, 25, 26, 27, 21, 22, 23, 17, 18, 19, 20, 14, 15, 16],
[21, 22, 23, 24, 25, 26, 27, 14, 15, 16, 17, 18, 19, 20, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 28, 29, 30, 31, 32, 33, 34],
[25, 26, 27, 21, 22, 23, 24, 18, 19, 20, 14, 15, 16, 17, 11, 12, 13, 7, 8, 9, 10, 4, 5, 6, 0, 1, 2, 3, 32, 33, 34, 28, 29, 30, 31],
[22, 23, 24, 25, 26, 27, 21, 15, 16, 17, 18, 19, 20, 14, 8, 9, 10, 11, 12, 13, 7, 1, 2, 3, 4, 5, 6, 0, 29, 30, 31, 32, 33, 34, 28],
[26, 27, 21, 22, 23, 24, 25, 19, 20, 14, 15, 16, 17, 18, 12, 13, 7, 8, 9, 10, 11, 5, 6, 0, 1, 2, 3, 4, 33, 34, 28, 29, 30, 31, 32],
[23, 24, 25, 26, 27, 21, 22, 16, 17, 18, 19, 20, 14, 15, 9, 10, 11, 12, 13, 7, 8, 2, 3, 4, 5, 6, 0, 1, 30, 31, 32, 33, 34, 28, 29],
[27, 21, 22, 23, 24, 25, 26, 20, 14, 15, 16, 17, 18, 19, 13, 7, 8, 9, 10, 11, 12, 6, 0, 1, 2, 3, 4, 5, 34, 28, 29, 30, 31, 32, 33],
[24, 25, 26, 27, 21, 22, 23, 17, 18, 19, 20, 14, 15, 16, 10, 11, 12, 13, 7, 8, 9, 3, 4, 5, 6, 0, 1, 2, 31, 32, 33, 34, 28, 29, 30]]

#[Shom8, Shom9, Shom10, Shom11, Shom12, Shom13].each {|tab| describe(tab); puts}

Corr = [0] + "1 5 11 12 7 2 4 10 6 3 8 9".split.map(&:to_i)

#if x = ARGV[0]
#  n = x.to_i
#  puts gen_cyclic_tptp(n)
#end

def test_cyclic i
  puts i
  File.write("1518cyclic.p", gen_cyclic_tptp(i))
  `~/misc/vampire -sa fmb -fmbss 1 -t 0 1518cyclic.p`
end

def test_all_cyclic m=1, n=5000
  trial = 2
  (m...n).each do |i|
    res = test_cyclic(i)
    puts res
    res_file = "1518cyclicresults#{trial}.txt"
    `touch #{res_file}`
    old = File.read(res_file)
    File.write(res_file, old + "\n\n" + res)
  end
end

def normal? x
  #normal form is a, aa, a(aa), a(a(aa)) i.e. left powers
  #x == A || (x[0] == A && normal?(x[1]))
  atom(x) || (atom(x[0]) && normal?(x[1]))
end

def show_form form, nested=false
  if atom(form)
    form.name
  else
    inner = "#{show_form(form[0], true)}.#{show_form(form[1], true)}"
    if nested
      "(#{inner})"
    else
      inner
    end
  end
end

class Element
  attr_accessor :forms

  def initialize *forms
    #forms are symbols or size-2 arrays of forms
    @forms = forms
  end

  def equate form
    #or take element?
    @forms << form unless eq?(form)
  end

  def normal_form
    @forms.find {|x| normal?(x)}
  end

  def form
    @forms[0]
  end

  def inspect
    "E[#{show_form(form)}]"
  end

  def to_s
    inspect
  end

  def copy
    Element.new(@forms)
  end

  def eq? x
    if x.is_a?(Element)
      !forms.intersection(x.forms).empty?
    else
      @forms.index(x)
    end
  end
end

def find_elt x, elts
  elts.find {|elt| elt.eq?(x)}
end

def ev_product x, y, elts
  #x and y elements
  #elts.find {|elt| elt.eq?([x.form, y.form])}
  find_elt([x.form, y.form], elts)
end

def atom x
  !x.is_a?(Array)
end

def simplify_expression expr, elts
  #an expression is a product (size 2 array) of expressions or a single element
  if atom(expr)
    expr
  else
    x, y = expr
    #puts "simp #{x.to_s}, #{y.to_s}, #{x.class} #{y.class}"
    if atom(x) && atom(y)
      #puts "atom #{x.to_s} #{y.to_s} #{x.class} #{y.class}"
      #puts "elts #{elts}"
      prod = ev_product(x, y, elts)
      #puts "prod #{prod}"
      ev_product(x, y, elts) || expr
    else
      #puts "nonatom #{x.to_s} / #{y.to_s}"
      x2 = simplify_expression(x, elts)
      y2 = simplify_expression(y, elts)
      #puts "simped #{x2.to_s} #{y2.to_s}"
      (atom(x2) && atom(y2) && ev_product(x2, y2, elts)) || [x2, y2]
    end
  end
end

def left_quotient x, y, elts
  #try to find z such that x = yz, which will be unique assuming that the magma operation is left-cancellative.
  elts.find {|z| x.eq?(ev_product(y, z, elts))}
end

def simplify_677 lhs, rhs, elts
  #x = y(x((yx)y))
  #the lhs and rhs may already be simplified using known information, we now try to simplify them further.
  rhs = simplify_expression(rhs, elts)
  #puts "677: #{lhs} = #{rhs}"
  #the lhs should always be an atom due to the form of 677
  if !atom(rhs) && atom(rhs[0])
    #here we cancel the variable on the left of the RHS, if we have enough information to do so.
    quot = left_quotient(lhs, rhs[0], elts)
    quot ? [quot, rhs[1]] : [lhs, rhs]
  else
    [lhs, rhs]
  end

  #yx = ev_product(y, x, elts)
  #if yx
  #  yx_y = ev_product(yx, y, elts)
  #  [y, [x, [yx, y]]]
  #else
  #[y, [x, [[y, x], y]]]
  ##give the simplest equality here
  #end
end

#def assume(elts, inequalities, i, x, y)#, steps: 10)
#  puts "trying #{elts[i].to_s} = #{x.to_s}*#{y.to_s}"
#  elts2 = elts.dup #new array, same elements to avoid copying
#  elts2[i] = Element.new(*(elts[i].forms + [[x.form, y.form]])) #normal_form?
#  return true
#end

#x = y(x((yx)y))
Rhs677 = -> x, y {[y, [x, [[y, x], y]]]}

Asym = :A
A = Element.new(Asym)

def cex_677_255 elts = [A], inequalities = [[A, [[[A, A], A], A]]], instances_677 = {}, hypothesis = nil
  #we assume a=1 is a counterexample to 255 in a finite 677 magma and try to deduce what the model looks like
  #we try setting products to existing elements and either derive a contradiction or leave it as a possibility.
  #a contradiction means either proving 255[a] or proving that known-distinct elements are equal.

  #x = ((xx)x)x = (x²x)x
  #rhs255 = -> x {[[[x, x], x], x]}

  #should we only add a new element when it's known to be distinct from all the others?
  while 1
    puts "elements: #{elts}"
    #puts "elements: #{elts.map(&:forms)}"
    #puts "a*a = #{ev_product(elts[0], elts[0], elts)}"
    #puts "a^2/a = #{left_quotient(elts[1]
    #construct and simplify 677 instances
    elts.product(elts).each do |x, y|
      i = [x.form, y.form]
      eq = instances_677[i] ? simplify_677(*instances_677[i], elts) : simplify_677(x, Rhs677[x, y], elts)
      instances_677[i] = eq
      #check that this doesn't equate known-distinct elements
      lhs = find_elt(eq[0], elts)
      rhs = find_elt(eq[1], elts)
      if lhs && rhs && lhs != rhs
        puts "equates #{lhs} and #{rhs}, contradiction"
        return false
      end
    end
    #p instances_677
    #need to refute a a^2 = a (2-cycle lemma)
    #Pf: if 
    inequalities.each do |lhs, rhs|
      #puts "checking #{lhs} != #{rhs}"
      sl = simplify_expression(lhs, elts)
      sr = simplify_expression(rhs, elts)
      if sl.eq?(sr)
        puts "ineq: #{lhs} now equals #{rhs}, contradiction"
        return false
      end
    end

    all_defined = true
    new_elt = false
    elts.product(elts).each do |x, y|
      if !ev_product(x, y, elts)
        all_defined = false #needed?
        prods = "#{x}.#{y}"
        puts "", "#{prods} undefined"
        refuted_all = true
        elts.each_with_index do |potential_prod, i|
          #try setting x*y equal to the ith element
          hyp = "#{x.to_s}*#{y.to_s} = #{elts[i].to_s}"
          puts "", "trying #{hyp}"
          elts2 = elts.dup #new array, same elements to avoid copying
          elts2[i] = Element.new(*(elts[i].forms + [[x.form, y.form]])) #normal_form?
          status = cex_677_255(elts2, inequalities.dup, instances_677.dup, hyp)

          if status
            puts "worked?"
            return status
            #refuted_all = false
            #continue to fill in model?
          else
            puts "refuted #{hyp}"
            inequalities << [elts[i], [x, y]]
          end
        end
        if refuted_all #the product cannot be an existing element
          if hypothesis #we're working under a hypothesis, so it's now refuted
            puts "no possible value for #{prods}, refuted #{hypothesis}"
            return false
          else #no hypotheses, add a new element for the product
            new_elt = Element.new([x.form, y.form])
            elts << new_elt
            break
          end
        end
      end
    end

    if new_elt
      puts "added new element: #{new_elt}, now have #{elts.size}"
    elsif all_defined
      puts "done"
      return elts
    else
      puts "continuing"
    end
    gets
  end
end

cex_677_255

#usage:
#describe(fmb_to_array("..."))
#renumber(table, Corr)
#./magma.rb 24 | vampire -sa fmb -fmbss 1 -t 0
