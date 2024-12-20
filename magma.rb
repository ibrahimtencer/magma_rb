#!/usr/bin/env ruby
require 'prime'

###############################
### General utility methods ###
###############################

def linear a, b, modulus
  #a linear magma operation in Z/(modulus)Z
  -> x, y {(a*x + b*y) % modulus}
end

def linear_table a, b, modulus
  #give the multiplication table for a linear magma
  f = linear(a, b, modulus)
  (0...modulus).map {|x| (0...modulus).map {|y| f[x, y]}}
end

def invert_perm perm
  #invert an array of integers seen as a permutation
  res = []
  perm.each_with_index.map {|x, i| res[x] = i}
  res
end

def inverse x, modulus
  #find the inverse of x mod modulus using brute force
  (1...modulus).each do |i|
    return i if (i * x) % modulus == 1
  end
end

def table_to_values table
  domain = interval(table.size)
  domain.product(domain).map {|i, j| [[i, j], table[i][j]]}.to_h
end

def interval n
  (0...n).to_a
end

def subset? ar1, ar2
  ar1.to_set.subset?(ar2.to_set)
end

def valid_table? table
  n = table.size
  domain = interval(n)
  table.all? {|row| row.is_a?(Array) && row.size == n && subset?(row, domain)}
end

def renumber table, ar
  #renumber the elements according to array of numbers ar
  #could also use the inverse permutation (ar[x], ar[y])
  vals = table_to_values(table)
  vals = vals.map {|(x, y), out| [[ar.index(x), ar.index(y)], ar.index(out)]}.to_h
  table_from_values(vals)
end

def column table, i
  table.map {|row| row[i]}
end

def transpose table
  #return the opposite operation: x *' y = y * x
  domain = interval(table.size)
  domain.map {|i| column(table, i)}
end

def injective? seq
  #interprets an array as a function on integers
  seq.uniq == seq
end

def satisfies?(table, &law)
  domain = interval(table.size)
  domain.product(domain).all?(&law)
end

def commutative? table
  domain = interval(table.size)
  domain.product(domain).all? {|x, y| table[x][y] == table[y][x]}
end

def associative? table
  domain = interval(table.size)
  domain.product(domain).product(domain).all? do |(x, y), z|
    table[x][table[y][z]] == table[table[x][y]][z]
  end
end

def left_cancellative? table
  domain = interval(table.size)
  domain.all? {|x| injective?(table[x])}
end

def right_cancellative? table
  #could be optimized
  left_cancellative?(transpose(table))
end

def identity table
  #left identity if it exists
  domain = interval(table.size)
  domain.find {|x| domain.all? {|y| table[x][y] == y}}
end

def inverses? table
  #does it have two-sided inverses?
  e = identity(table)
  if e
    #return the identity if it has inverses
    domain = interval(table.size)
    domain.all? {|x| domain.any? {|y| table[x][y] == e && table[y][x] == e}} && e
  else
    puts "no identity"
  end
end

def homomorphism? seq, table
  #seq is a function given as an array of numbers, we return whether it is a homomorphism wrt the multiplication table
  domain = interval(table.size)
  domain.product(domain).all? {|x, y| seq[table[x][y]] == table[seq[x]][seq[y]]}
end

def idempotent? table
  interval(table.size).all? {|i| table[i][i] == i}
end

def fixpoints seq
  seq.each_with_index.select {|x, i| x == i}
end

def unique_fixpoint_of_L? table
  table.all? {|row| fixpoints(row).size == 1}
end

Expx ||= -> f, x, y {x}

Equations ||=
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
  #whether b is a candidate for the second coefficient in a linear magma
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
  #find candidates for the second coefficient in a linear magma using brute force
  (0...modulus).select {|n| b_candidate?(eqn, n, modulus)}
end

def check op, eqn, lhs_fn, rhs_fn, x, y
  lhs = lhs_fn[op, x, y]
  rhs = rhs_fn[op, x, y]
  if lhs != rhs
    p [x, y, lhs, rhs, "#{eqn} failed"]
  end
end

def calc_a eqn, b, modulus
  #calculate a from b
  case eqn
  when 1518
    b_inv = inverse(b, modulus) #(modulus + 1)/b
    ((1 - b*b*b) * b_inv) % modulus
  when 677
    #a = 1/b(1 + b²)
    (inverse(b, modulus) * inverse(1 + b*b, modulus)) % modulus
  end
end

$none = []

def search_linear antecedent = 677, cons = [255], upto: 20, describe: false, describe_op: false
  #generate linear models for an equation
  #cons is the consequents
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

      table = linear_table(a, b, m)
      describe(table) if describe
      describe(transpose(table)) if describe_op
    end
  end
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
  #output a multiplication table as a square grid
  puts "[#{ar.map(&:to_s).join(",\n")}]"
  ar
end

def nice_fmb_to_array str
  line_break_array(fmb_to_array(str))
end

def cycles perm, show_fixpoints=false
  #gives the cycles of a permutation (array of numbers in 0...len), as an array of arrays
  #will go into an infinite loop if it's not a bijection
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

def linear_auxiliary table, &blk
  #Assume that the magma operation is something like a cancellative linear one and calculate the addition operation in the original linear structure associated with the magma.
  #More specifically we assume that we have a cancellative magma N with identity 0 and two invertible functions that preserve 0, and that the operation . on M is defined as
  #  x.y = f(x) + g(y)
  #Then M will be cancellative and 0 will be idempotent in M.
  #We recover the operation + from M via
  #  (R₀⁻¹x).(L₀⁻¹y) = f(f⁻¹(x)) + g(g⁻¹(y)) = x + y
  #and determine whether it is associative and commutative.

  #we take a block for processing because some of the magmas are huge and also idempotent so they would generate an auxiliary magma for every element

  domain = interval(table.size)
  puts "size: #{table.size}"
  #the zero must be idempotent, and its left and right multiplication operations must be invertible for this construction to work
  poss_zeros = domain.select {|i| injective?(table[i]) && injective?(column(table, i)) && table[i][i] == i}
  if poss_zeros.empty?
    #puts "no possible zeros"
    return nil
  #else
  #  puts "trying possible zeros (#{poss_zeros.size}): #{poss_zeros}"
  end
  processor = blk || -> x {x}
  poss_zeros.map do |z|
    #left and right multiplication by z:
    l_z = table[z]
    r_z = column(table, z)
    l_z_inv = invert_perm(l_z)
    r_z_inv = invert_perm(r_z)
    #puts "L_z^-1: #{l_z_inv}"
    #puts "R_z^-1: #{r_z_inv}"

    #x + y = (Rₑ⁻¹x).(Lₑ⁻¹y)
    #give the table for + along with the two functions
    tab = domain.map {|x| domain.map {|y| table[r_z_inv[x]][l_z_inv[y]]}}
    blk[tab, l_z, r_z]
  end
end

def parse_extensions
  tables = []
  current_table = []
  #346 tables
  File.new("677_probably_nonlinear.txt").each_line do |line|
    if line =~ /\A(\d+\s*)+\z/
    #if line =~ /\A(\d+\s+)+\d+\z/
      is = line.split.map(&:to_i)
      current_table << is if is.size > 1
    elsif line =~ /=+/
      tables << current_table
      current_table = []
    end
  end

  tables
end

def analyze_extensions
  #analyzes cohomological extensions of linear 677 magmas - are they linear?
  #tns = 0...(tables.size)
  #takes ~20 min
  tables = parse_extensions()
  puts "#{tables.size} tables"
  puts "sizes: #{tables.map(&:size)}"
  puts "all valid" if tables.all? {|t| valid_table?(t)}
  puts "all left cancellative" if tables.all? {|t| left_cancellative?(t)}
  puts "all right cancellative" if tables.all? {|t| right_cancellative?(t)}
  tables.map do |table|
    res = linear_auxiliary(table)

    if res
      res.map do |tab, r, l|
        assoc = associative?(tab)
        comm = commutative?(tab)
        if assoc && comm
          :ac
        elsif assoc
          :a
        elsif comm
          :c
        else
          :n
        end
      end.tally
    else
      :nz
    end
  end
end

def describe table, show_fixpoints=false
  #describe the multiplication table for a 1518 or other magma (but works best when L_x is bijective and there are repeated L_x's)
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

#[Shom8, Shom9, Shom10, Shom11, Shom12, Shom13].each {|tab| describe(tab); puts}

#Corr = [0] + "1 5 11 12 7 2 4 10 6 3 8 9".split.map(&:to_i)

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


#########################################
### FINITE MODEL BUILDER FOR 677=>255 ###
#########################################

def atom x
  !x.is_a?(Array)
end

def normal? x
  #normal form is a, aa, a(aa), a(a(aa)) i.e. left powers
  #x == A || (x[0] == A && normal?(x[1]))
  atom(x) || (atom(x[0]) && normal?(x[1]))
end

def show_normal_form form
  #we name left powers of A as A, B, C, D...
  if atom(form)
    "A"
  else
    show_normal_form(form[1]).succ
  end
end

def show_form form, nested=false
  if normal?(form)
    show_normal_form(form)
  elsif atom(form)
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
  #evaluate the product x*y, x and y elements
  #elts.find {|elt| elt.eq?([x.form, y.form])}
  find_elt([x.form, y.form], elts)
end

def simplify_expression expr, elts
  #an expression is a product (size 2 array) of expressions or a single element
  if atom(expr)
    expr
  else
    x, y = expr
    if atom(x) && atom(y)
      prod = ev_product(x, y, elts)
      ev_product(x, y, elts) || expr
    else
      x2 = simplify_expression(x, elts)
      y2 = simplify_expression(y, elts)
      (atom(x2) && atom(y2) && ev_product(x2, y2, elts)) || [x2, y2]
    end
  end
end

def left_quotient x, y, elts
  #try to find z such that x = yz, which will be unique assuming that the magma operation is left-cancellative.
  elts.find {|z| x == ev_product(y, z, elts)} #was eq?
end

def simplify_677 lhs, rhs, elts
  #x = y(x((yx)y))
  #We take an instance of 677 and try to simplify it using known products and left-cancellativity.
  #The lhs and rhs may already be simplified, we try to simplify them further.
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

def multiplication_table elts
  #the multiplication table so far (with nils for undefined products)
  elts.each_with_index.map do |x, i|
    elts.each_with_index.map do |y, j|
      prod = ev_product(x, y, elts)
      prod && elts.index(prod)
    end
  end
end

def show_instances instances, elts
  #print the instances of 677
  instances.each {|k, v| puts "#{find_elt(k[0], elts)}, #{find_elt(k[1], elts)} : #{v[0]} = #{v[1]}"}
end

#def assume(elts, inequalities, i, x, y)#, steps: 10)
#  puts "trying #{elts[i].to_s} = #{x.to_s}*#{y.to_s}"
#  elts2 = elts.dup #new array, same elements to avoid copying
#  elts2[i] = Element.new(*(elts[i].forms + [[x.form, y.form]])) #normal_form?
#  return true
#end

#x = y(x((yx)y))
Rhs677 ||= -> x, y {[y, [x, [[y, x], y]]]}

Asym ||= :A
A ||= Element.new(Asym)
LEVEL ||= 6 #the cutoff to do logging

def cex_677_255 elts = [A], inequalities = [[A, [[[A, A], A], A]]], instances_677 = {}, hypothesis = nil, level = 0
  #We assume A is a counterexample to 255 in a finite 677 magma and try to deduce what the model looks like.
  #We try setting products to existing elements and either derive a contradiction or leave it as a possibility.
  #A contradiction means either proving 255[a] or proving that known-distinct elements are equal.
  #Currently we assume the model has A as generator and also consists of the first n left powers of A, so this is not an exhaustive search of any order past 2. But we could add new elements in other ways.

  #x = ((xx)x)x = (x²x)x
  #rhs255 = -> x {[[[x, x], x], x]}

  while 1
    #puts "elements: #{elts}"
    #construct and simplify 677 instances
    elts.product(elts).each do |x, y|
      i = [x.form, y.form]
      eq = instances_677[i] ? simplify_677(*instances_677[i], elts) : simplify_677(x, Rhs677[x, y], elts)
      instances_677[i] = eq
      #check that this doesn't equate known-distinct elements
      lhs = find_elt(eq[0], elts)
      rhs = find_elt(eq[1], elts)
      if lhs && rhs && lhs != rhs
        #puts "equates #{lhs} and #{rhs}, contradiction"
        return false
      end
    end
    #p instances_677
    inequalities.each do |lhs, rhs|
      #puts "checking #{lhs} != #{rhs}"
      sl = simplify_expression(lhs, elts)
      sr = simplify_expression(rhs, elts)
      if sl.eq?(sr)
        #puts "ineq: #{lhs} now equals #{rhs}, contradiction"
        return false
      end
    end

    all_defined = true
    new_elt = false
    elts.product(elts).each do |x, y|
      if !ev_product(x, y, elts)
        all_defined = false #needed?
        refuted_all = true
        #ensure left-cancellativity
        possible_prods = elts - (elts - [y]).map {|elt| ev_product(x, elt, elts)}
        if level < LEVEL
          prods = "#{x}.#{y}"
          puts("", "#{prods} undefined, trying to define (level #{level})")
          show_instances(instances_677, elts)
          puts "possible: #{possible_prods}"
          puts line_break_array(multiplication_table(elts))
        end
        possible_prods.each_with_index do |possible_prod|
          i = elts.index(possible_prod)
          #try setting x*y equal to the ith element
          hyp = "#{x.to_s}*#{y.to_s} = #{elts[i].to_s}"
          #puts "", "trying #{hyp}"
          elts2 = elts.dup #new array, same elements to avoid copying
          elts2[i] = Element.new(*(elts[i].forms + [[x.form, y.form]])) #normal_form?
          status = cex_677_255(elts2, inequalities.dup, instances_677.dup, hyp, level + 1)

          if status
            puts "worked?"
            return status
            #refuted_all = false
            #continue to fill in model?
          else
            #puts "refuted #{hyp}"
            inequalities << [elts[i], [x, y]]
          end
        end
        if refuted_all #the product cannot be an existing element
          if hypothesis #we're working under a hypothesis, so it's now refuted
            if level < LEVEL
              #puts line_break_array(multiplication_table(elts))
              show_instances(instances_677, elts)
              puts "no possible value for #{prods}, refuted #{hypothesis} (level #{level})"
            end
            return false
          else #no hypotheses, add a new element for the product
            new_elt = Element.new([x.form, y.form])
            puts line_break_array(multiplication_table(elts))
            elts << new_elt
            break
          end
        end
      end
    end

    puts; puts
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


#usage:
#describe(fmb_to_array("..."))
#renumber(table, Corr)
#./magma.rb 24 | vampire -sa fmb -fmbss 1 -t 0

#tabs = parse_extensions; Pseudolinear677 = [tabs[0], tabs[98], tabs[240]] #also in magma_tables.rb
#Pseudolinear677.each {|m| linear_auxiliary(m) {|t, l, r| puts associative?(t); puts commutative?(t); puts homomorphism?(r, t); puts homomorphism?(l, t); puts}}
