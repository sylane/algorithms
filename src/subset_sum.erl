%% ===========================================================================
%% @doc Search a list of integers for a subset summing zero. 
%% 
%% == Requirements ==
%%
%% <ul>
%%  <li>Should be fast, very fast.</li>
%%  <li>Should find ONE subset summing zero.</li>
%%  <li>Should work with BIG lists of integers (> 1000000).</li>
%%  <li>Should be parallelizable.</li>
%%  <li>Should not take exponential amounts of time.</li>
%%  <li>Should not take exponential amounts of memory.</li>
%% </ul>
%%
%%
%% == Assumptions ==
%%
%% <ul>
%%  <li>The set of integer is a uniform distribution.</li>
%%  <li>The range of the integer is small and centered on zero,
%%      something like  [-65000..65000].</li>
%%  <li>False negative result are acceptable.</li>
%% </ul>
%%
%% == Algorithm ==
%%
%% Because the problem is NP complete we cannot possibly resolve the problem
%% by testing all combinations so we have to create a good enough
%% approximation. So we reduce the solution space with some limitations:
%%
%% <ul>
%%  <li>Limit to subsets up to 32 integers.</li>
%%  <li>Limit to subsets coming from two partitions of 16 integers.</li>
%% </ul>
%%
%% With these limitations, we know that for sets of up to 32 integers we have
%% the guarantee to find a subset summing zero if it exists. For bigger sets
%% of integers, only part of the combination space will be explored but
%% because we are not interested in all the solutions, and the probabilities
%% are very high for uniform distribution of integer it is good enough.
%%
%% The algorithm has two phases:
%%
%% === Phase 1: Expand ===
%%
%% The set of value is successively combined and populated with intermediary
%% results checking for subset summing zero on the way: 
%%
%% `[A, B, C, D, E, F, G, H |Etc]'
%%
%% <ol>
%%  <li>`[[A, B, A+B], [C, D, C+D], [E, F, E+F], [G, H, G+H] |Etc]'</li>
%%  <li>`[[A, B, A+B, C, D, C+D, A+C, A+D, A+C+D, B+C, B+D, B+C+D, A+B+C, A+B+D. A+B+C+D],
%%       [E, F, E+F, G, H, G+H, E+G, E+H, E+G+H, F+G, F+H, F+G+H, E+F+G, E+F+H, E+F+G+H] |Etc]'</li>
%%  <li>`...'</li>
%% </ol>
%%
%% Because we don't want the memory usage to grow exponentially this step is
%% done only 4 times to stay in an acceptable range of memory consumption.
%% After this step we have all sums up to 16 integers for every
%% partitions of 16 integers.
%% 
%%
%% === Phase 2: Combinate ===
%%
%% Each pairs of intermediary results are tagged, merged and sorted using the
%% tag and the absolute value as key. Then the result is browsed only one time
%% for contiguous entries with different tags summing zero.
%%
%% With this second step we found all subsets up to 32 integers coming from
%% up to 2 partitions of 16 integers. 
%%
%%
%% == Parallelization ==
%%
%% The algorithm could easily be parallelized an processed by different
%% processes, by splitting the list in phase 1 and distributing block
%% processing in phase 2. 
%%
%%
%% == Usage ==
%%
%% <pre>
%% $ make shell 
%% 1> L = subset_sum:random_data(1000).
%% 2> subset_sum:find_subset(L).
%% </pre>
%%
%%
%% == Benchmark ==
%%
%% <pre>
%% $ make shell
%% 1> subset_sum:benchmark(1000, 20).
%% </pre>
%%
%%
%% @since      Feb 25, 2012
%% @version    1.0
%% @copyright  2012, Sebastien Merle <s.merle@gmail.com>
%% @author     Sebastien Merle <s.merle@gmail.com>
%% @end
%%
%% Copyright (c) 2012, Sebastien Merle <s.merle@gmail.com>
%% All rights reserved.
%%
%% Redistribution and use in source and binary forms, with or without
%% modification, are permitted provided that the following conditions are met:
%%
%%   * Redistributions of source code must retain the above copyright
%%     notice, this list of conditions and the following disclaimer.
%%   * Redistributions in binary form must reproduce the above copyright
%%     notice, this list of conditions and the following disclaimer in the
%%     documentation and/or other materials provided with the distribution.
%%
%% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
%% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
%% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
%% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
%% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
%% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
%% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
%% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
%% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%% POSSIBILITY OF SUCH DAMAGE.
%% ===========================================================================

-module(subset_sum).

-author('Sebastien Merle <s.merle@gmail.com>').


%% --------------------------------------------------------------------
%% Exports
%% --------------------------------------------------------------------

%% API exports
-export([find_subset/1,
         has_subset/1,
         random_data/1,
         benchmark/2]).


%% --------------------------------------------------------------------
%% Macros
%% --------------------------------------------------------------------

-define(MERGE_ITERATIONS, 4).


%% --------------------------------------------------------------------
%% Types
%% --------------------------------------------------------------------

-type neg_or_zero() :: neg_integer() | 0. 

-type pos_or_zero() :: pos_integer() | 0.

-type value() :: integer().

-type value_set() :: list(value()) | [].

-type struct() :: {value(), list(value)}.

-type struct_set() :: list(struct()) | [].

-type struct_sets() :: list(struct_set()) | [].

-type tag() :: atom().

-type tagged() :: {tag(), value(), list(value)}.

-type tagged_set() :: list(tagged()) | [].


%% --------------------------------------------------------------------
%% API Functions
%% --------------------------------------------------------------------

%% --------------------------------------------------------------------
%% @spec find_subset(Set::value_set()) -> not_found | value_set()
%%   where value_set() = [] | [integer()]
%%
%% @doc Returns a subset of the specified set of integer whose sum gives zero,
%% or 'not_found' if no such a subset has been found. Up to sets of 32
%% integers, if 'not_found' is returned it is guaranteed that the specified
%% set do not contains any subset summing zero. For bigger sets, it only means
%% that the approximation algorithm did not find any.

-spec find_subset(Set::value_set()) -> not_found | value_set().

find_subset(Set) -> search_(Set).


%% --------------------------------------------------------------------
%% @spec has_subset(Set::value_set()) -> boolean()
%%   where value_set() = [] | [integer()]
%%
%% @doc Returns 'true' if the specified set of integer contains a subset
%% summing zero, of 'false' if not found. Up to sets of 32 integers,
%% if 'false' is returned it is guaranteed that the specified set do not
%% contains any subset summing zero. For bigger sets it only means that the
%% approximation algorithm did not find any.

-spec has_subset(Set::value_set()) -> boolean().

has_subset(Set) ->
    case search_(Set) of
        not_found -> false;
        [_|_] -> true
    end.


%% --------------------------------------------------------------------
%% @spec random_data(Size::pos_integer()) -> Set::value_set()
%%   where value_set() = [] | [integer()]
%%
%% @doc Returns a list of non-zero integers in range [-65000..65000]
%% of specified size using an uniform random distribution.

-spec random_data(Size::pos_integer()) -> value_set().

random_data(Size) -> random_data_(Size).


%% --------------------------------------------------------------------
%% @spec benchmark(Size::pos_integer(), Iter::pos_integer()) -> ok
%%
%% @doc Benchmarks find_subset/1 with a random list of integers of specified
%% size multiple times and print the result to the console. 

-spec benchmark(Size::pos_integer(), Iter::pos_integer()) -> ok.

benchmark(Size, Iter) -> benchmark_(Size, Iter, 0, [], 0).


%% --------------------------------------------------------------------
%% Internal Functions
%% --------------------------------------------------------------------

%% --------------------------------------------------------------------
%% Searches the specified set of integer for a subset summing zero.
%% Returns such a subset if found or 'not_found' otherwise.

-spec search_(Set::value_set())
        -> not_found | value_set().

search_(Set) ->
    {Min, Max, Sets} = prepare_(Set),
    search_(Min, Max, Sets).


-spec search_(Min::integer(), Max::integer(), Sets::struct_sets())
        -> not_found | value_set().

search_(Min, Max, _Sets) when Min >= 0; Max =< 0 -> not_found;
search_(_Min, _Max, Sets) ->
    case merge_(Sets, ?MERGE_ITERATIONS) of
        {found, Result} -> Result;
        NewSets -> check_(NewSets)
    end.


%% --------------------------------------------------------------------
%% Phase 1 successively merges the sets by computing all possible sums,
%% checking the results for zero and populating the the sets with them.
%% A maximum number of iteration should be specified to not consume all
%% the system memory. Returns 'found' with the set of integer summing to zero
%% if any is found or the merged sets.

-spec merge_(Sets::struct_sets(), Iter::pos_integer()) ->
          {found, value_set()} | struct_sets().

merge_([_] = Sets, _Iter) -> Sets;
merge_(Sets, 0) -> Sets;
merge_(Sets, Iter) ->
    case expand_(Sets) of
        {found, Result} -> {found, Result};
        NewSets -> merge_(NewSets, Iter - 1)
    end.


%% --------------------------------------------------------------------
%% Phase 2 checks every pairs of integer list if the sum of two elements
%% from a different list gives zero. To be faster than O(n^2) the two lists
%% are tagged and merged, then the result is sorted by tag and absolute value.
%% The valued are checked sequentially for pairs of values with different
%% tag summing zero.
%% This way the complexity should be the one of the sort algorithm. 

-spec check_(Sets::struct_sets()) -> not_found | value_set().

check_([]) -> not_found;
check_([_]) -> not_found;
check_([_H |T] = All) -> check_all_(All, T).


-spec check_all_(As::struct_sets(), Bs::struct_sets()) ->
          not_found | value_set().

check_all_([], _Bs) -> not_found;
check_all_([_], []) -> not_found;
check_all_([_A1, A2 |As], []) -> check_all_([A2 |As], As);
check_all_([A |_] = As, [B |Bs]) ->
    case check_blocks_(A, B) of
        not_found -> check_all_(As, Bs);
        Result -> Result
    end.


-spec check_blocks_(A::struct_set(), B::struct_set()) -> not_found | value_set().

check_blocks_(A, B) ->
    L = tag_(B, b, tag_(A, a, [])),
    S = lists:sort(fun predicate_/2, L),
    check_sorted_(S).


-spec tag_(Set::struct_set(), Tag::tag(), Acc::tagged_set()) -> tagged_set().

tag_([], _T, Acc) -> Acc;
tag_([{V, E} |Vs], T, Acc) -> tag_(Vs, T, [{T, V, E} |Acc]).


-spec predicate_(A::tagged(), B::tagged()) -> boolean().

predicate_(A, B) ->
    T1 = {abs(element(2, A)), element(1, A)},
    T2 = {abs(element(2, B)), element(1, B)},
    T1 =< T2.


-spec check_sorted_(L::tagged_set()) -> not_found | value_set().

check_sorted_([]) -> not_found;
check_sorted_([_]) -> not_found;
check_sorted_([{T, _, _}, {T, _, _} = B |Rem]) ->
    check_sorted_([B |Rem]);
check_sorted_([{_, Av, Ae}, {_, Bv, Be} |_Rem])
    when Av + Bv =:= 0 -> Ae ++ Be;
check_sorted_([_A |Rem]) ->
    check_sorted_(Rem).


%% --------------------------------------------------------------------
%% Prepares the algorithm by calculating the biggest and smallest
%% possible sum values and encapsulating each value in the structure
%% needed by the algorithm.
%% [A, B, C] -> [[{A, [A]}], [{B, [B]}], [{C, [C]}]]

-spec prepare_(Set::value_set()) ->
          {neg_or_zero(), pos_or_zero(), struct_sets()}.

prepare_(Set) -> prepare_(Set, 0, 0, []).


-spec prepare_(Set::value_set(), Neg::neg_or_zero(),
               Pos::pos_or_zero(), Result::struct_sets()) ->
          {neg_or_zero(), pos_or_zero(), struct_sets()}.

prepare_([], Neg, Pos, Result) -> {Neg, Pos, Result};
prepare_([V |Set], Neg, Pos, Result) when V >= 0 ->
    prepare_(Set, Neg, Pos + V, [[{V, [V]}] |Result]);
prepare_([V |Set], Neg, Pos, Result) ->
    prepare_(Set, Neg + V, Pos, [[{V, [V]}] |Result]).


%% --------------------------------------------------------------------
%% Expends the set of value by combining every pairs of subsets
%% into a single on containing all the sum combinations.
%% Returns 'found' with the set of integer summing zero if found,
%% or the list of combined sets if not found.

-spec expand_(Sets::struct_sets()) -> {found, value_set()} | struct_sets().

expand_(Sets) -> expand_(Sets, []).


-spec expand_(Sets::struct_sets(), Acc::struct_sets()) ->
          {found, value_set()} | struct_sets().

expand_([], Acc) -> Acc;
expand_([A], Acc) -> [A |Acc];
expand_([A, B |Sets], Acc) ->
    case combinate_(A, B) of
        {found, Result} -> {found, Result};
        SubSet -> expand_(Sets, [SubSet |Acc])
    end.
    

%% --------------------------------------------------------------------
%% Combines two set of values into a new set containing all the values of
%% each sets plus all possible sums of two elements of different sets.
%% Returns 'found' and the set of integer summing zero if found or
%% the combined set if not.

-spec combinate_(As::struct_set(), Bs::struct_set()) ->
          {found, value_set()} | struct_set().

combinate_(As, Bs) -> combinate_(As, Bs, As ++ Bs).


-spec combinate_(As::struct_set(), Bs::struct_set(), Acc::struct_set()) ->
          {found, value_set()} | struct_set().

combinate_([], _Bs, Acc) -> Acc;
combinate_([A |As], Bs, Acc) -> combinate_(As, Bs, A, Bs, Acc).


-spec combinate_(As::struct_set(), AllBs::struct_set(),
                 A::struct(), Bs::struct_set(), Acc::struct_set()) ->
          {found, value_set()} | struct_set().

combinate_(As, AllBs, _A, [], Acc) -> combinate_(As, AllBs, Acc);
combinate_(_As, _AllBs, {Av, Ac}, [{Bv, Bc} |_Bs], _Acc)
  when Av + Bv =:= 0 -> {found, Ac ++ Bc};
combinate_(As, AllBs, {Av, Ac} = A, [{Bv, Bc} |Bs], Acc) ->
    combinate_(As, AllBs, A, Bs, [{Av + Bv, Ac ++ Bc} |Acc]).


%% --------------------------------------------------------------------
%% Generates a list of non-zero integers in the range [-65000..65000]
%% using an uniform random distribution. 

-spec random_data_(Size::pos_integer()) -> value_set().

random_data_(Size) ->
    random_data_(Size, random_value_(), []).


-spec random_data_(Rem::pos_integer(), Value::integer(),
                      Acc::value_set()) ->
          value_set().

random_data_(0, _Value, Acc) -> Acc;
random_data_(Rem, 0, Acc) ->
    random_data_(Rem, random_value_(), Acc);
random_data_(Rem, Value, Acc) ->
    random_data_(Rem - 1, random_value_(), [Value |Acc]).


-spec random_value_() -> integer().

random_value_() ->
    random:uniform(65000*2+1)-65000-1.


%% --------------------------------------------------------------------
%% Benchmarks find_subset/1 with lists of random integers of specified
%% size, and print the result to the console.

-spec benchmark_(Size::pos_integer(), Iter::pos_integer(),
                 Length::pos_or_zero(), Samples::list(pos_or_zero()),
                 Found::pos_or_zero()) -> ok.

benchmark_(_Size, Length, Length, Samples, Found) ->
    Min = lists:min(Samples),
    Max = lists:max(Samples),
    Med = lists:nth(round((Length / 2)), lists:sort(Samples)),
    Avg = round(lists:foldl(fun(X, Sum) -> X + Sum end, 0, Samples) / Length),
    Rate = round(Found * 100 / Length),
    Sep = string:copies("*", 21),
    io:format("~s~n" 
              "Min:     ~9b 탎~n"
              "Max:     ~9b 탎~n"
              "Median:  ~9b 탎~n"
              "Average: ~9b 탎~n"
              "Success: ~9b %~n"
              "~s~n", [Sep, Min, Max, Med, Avg, Rate, Sep]);
benchmark_(Size, Iter, Count, Samples, Found) ->
    io:format("Round ~3b...", [Count + 1]),
    Set = random_data(Size),
    case timer:tc(fun find_subset/1, [Set]) of
        {Time, not_found} ->
            io:format("  not found~n"),
            benchmark_(Size, Iter, Count + 1, [Time |Samples], Found);
        {Time, [_|_] = Result} ->
            io:format(" ~w~n", [Result]),
            benchmark_(Size, Iter, Count + 1, [Time |Samples], Found + 1)
    end.
    

%% --------------------------------------------------------------------
%% Tests
%% --------------------------------------------------------------------

-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").

prepare_test() ->
    ?assertMatch({0, 0, []}, prepare_([])),
    ?assertMatch({0, 6, [[{3, [3]}], [{2, [2]}], [{1, [1]}]]},
                 prepare_([1, 2, 3])),
    ?assertMatch({-8, 11, [[{8, [8]}], [{-3, [-3]}], [{2, [2]}],
                           [{-5, [-5]}], [{1, [1]}]]},
                 prepare_([1, -5, 2, -3, 8])),
    ok.

expand_test() ->
    ?assertMatch([], expand_([])),
    ?assertMatch([[{1, [1]}]], expand_([[{1, [1]}]])),
    ?assertMatch([[{3, [1, 2]}, {1, [1]}, {2, [2]}]],
                 expand_([[{1, [1]}], [{2, [2]}]])),
    ?assertMatch([[{3, [3]}], [{3, [1, 2]}, {1, [1]}, {2, [2]}]],
                 expand_([[{1, [1]}], [{2, [2]}], [{3, [3]}]])),
    ?assertMatch([[{7, [3, 4]}, {3, [3]}, {4, [4]}],
                  [{3, [1, 2]}, {1, [1]}, {2, [2]}]],
                 expand_([[{1, [1]}], [{2, [2]}], [{3, [3]}], [{4, [4]}]])),
    
    ?assertMatch([[{8, [3, 3, 2]}, {7, [3, 4]}, {7, [2, 3, 2]}, {6, [2, 4]},
                   {6, [1, 3, 2]}, {5, [1, 4]}, {1, [1]},
                   {2, [2]}, {3, [3]}, {4, [4]}, {5, [3, 2]}]],
                 expand_([[{1, [1]}, {2, [2]}, {3, [3]}],
                          [{4, [4]}, {5, [3, 2]}]])),
    
    ?assertMatch({found, [-5, 5]}, expand_([[{-5, [-5]}], [{5, [5]}]])),
    ok.

combinate_test() ->
    ?assertMatch([{3, [1, 2]}, {1, [1]}, {2, [2]}],
                 combinate_([{1, [1]}], [{2, [2]}])),
    ?assertMatch([{5, [2, 3]}, {4, [1, 3]},
                  {1, [1]}, {2, [2]}, {3, [3]}],
                 combinate_([{1, [1]}, {2, [2]}], [{3, [3]}])),
    ?assertMatch([{4, [1, 3]}, {3, [1, 2]},
                  {1, [1]}, {2, [2]}, {3, [3]}],
                 combinate_([{1, [1]}], [{2, [2]}, {3, [3]}])),
    ?assertMatch([{6, [2, 4]}, {5, [2, 3]}, {5, [1, 4]}, {4, [1, 3]},
                  {1, [1]}, {2, [2]}, {3, [3]}, {4, [4]}],
                 combinate_([{1, [1]}, {2, [2]}], [{3, [3]}, {4, [4]}])),
    ok.

check_test() ->
    ?assertMatch(not_found, check_([])),
    ?assertMatch(not_found, check_([[{-5, [-5]}, {5, [5]}]])),
    ?assertMatch(not_found, check_([[{-5, [-5]}, {5, [5]}], []])),
    ?assertMatch([a, b, c, d],
                 check_([[{-5, [a, b]}], [{5, [c, d]}]])),
    ?assertMatch([a, f],
                 check_([[{3, [a]}, {4, [b]}],
                         [{8, [c]}, {7, [d]}],
                         [{2, [e]}, {-3, [f]}]])),
    ok.

search_test() ->
    ?assertMatch([1, 0, -3, 2], search_([0, 1, 2, -3])),
    ?assertMatch([-3, 2, 1], search_([1, 4, 5, 2, -3])),
    ?assertMatch([1, 2, -3], search_([1, 2, 3, 4, 5, 6, 7, 8, 9,
                                      10, 11, 12, 13, 14, -3])),
    
    ?assertMatch(not_found, search_([1, 2, 3, -8])),
    ?assertMatch(not_found, search_([1, 4, 5, 2])),
    ?assertMatch(not_found, search_([-1, -5, -8])),
    ok.

-endif.
