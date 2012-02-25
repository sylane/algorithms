%% ===========================================================================
%% @doc Find the closest pair of 2D points from a given set of points. 
%%
%% This is an Erlang implementation of a well known recursive divide and
%% conquer algorithm with complexity O(n log n).
%%
%% == Usage ==
%%
%% <pre>
%% $ make shell
%% 1> D = closest_points:random_data(10000).
%% 2> closest_points:find_closest(D).
%% </pre>
%% 
%%
%% == Benchmark ==
%%
%% <pre>
%% $ make shell
%% 1> closest_points:benchmark(10000, 20).
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

-module(closest_points).

-author('Sebastien Merle <s.merle@gmail.com>').


%% --------------------------------------------------------------------
%% Exports
%% --------------------------------------------------------------------

%% API exports
-export([find_closest/1,
         random_data/1,
         benchmark/2]).


%% --------------------------------------------------------------------
%% Types
%% --------------------------------------------------------------------

-type pos_or_zero() :: pos_integer() | 0.

-type point() :: {integer(), integer()}.

-type points() :: list(point()) | [].

-type pair() :: {point(), point()}.

-type result() :: {number(), pair()}.


%% --------------------------------------------------------------------
%% API Functions
%% --------------------------------------------------------------------

%% --------------------------------------------------------------------
%% @spec find_closest(Points::points()) -> result() | undefined
%%   where points() = list(point())
%%         result() = {number(), pair()}
%%         pair() = {point(), point()}
%%         point() = {integer(), integer()}
%%
%% @doc Returns the closest pair of points from the specified set.

-spec find_closest(Points::points()) -> result() | undefined.

find_closest(Points) -> closest_(Points).


%% --------------------------------------------------------------------
%% @spec random_data(Size::pos_integer()) -> points()
%%   where points() = list(point())
%%         point() = {integer(), integer()}
%%
%% @doc Returns a random list of 2D points with coordinates in the
%% range [-65000..65000].

-spec random_data(Size::pos_integer()) -> points().

random_data(Size) -> random_data_(Size, []).


%% --------------------------------------------------------------------
%% @spec benchmark(Size::pos_integer(), Iter::pos_integer()) -> ok
%%
%% @doc Benchmarks find_closest/1 multiple times with a list of specified size
%% filled with random points generated by random_data/1.

-spec benchmark(Size::pos_integer(), Iter::pos_integer()) -> ok.

benchmark(Size, Iter) -> benchmark_(fun find_closest/1, Size, Iter, 0, []).


%% --------------------------------------------------------------------
%% Internal Functions
%% --------------------------------------------------------------------

%% --------------------------------------------------------------------
%% Searches a list of 2D points for the closest pair using brute force.

-spec brute_force_(Points::points()) -> result() | undefined.

brute_force_([]) -> undefined;
brute_force_([_]) -> undefined;
brute_force_([A, B |Rem] = All) ->
    brute_force_(All, Rem, {distance_(A, B), {A, B}}).


-spec brute_force_(As::points(), Bs::points(), Result::result()) -> result().

brute_force_([_], [], Result) -> Result;
brute_force_([_A1, A2 |As], [], Result) ->
    brute_force_([A2 |As], As, Result);
brute_force_([A |_] = As, [B |Bs], Result) ->
    NewResult = select_(Result, {distance_(A, B), {A, B}}),
    brute_force_(As, Bs, NewResult).


%% --------------------------------------------------------------------
%% Searches a list of 2D points for the closest pair unsing a recursive
%% divide and conquer algorithm with complexity O(n log n).

-spec closest_(Points::points()) -> result() | undefined.

closest_(Points) ->
    Size = length(Points),
    Px = lists:keysort(1, Points),
    Py = lists:keysort(2, Points),
    divide_(Size, Px, Py).


%% Divide the space vertically, recurse to find the closest pair in each ones.
%% Px should be ordered on X and Py ordered on Y.

-spec divide_(Size::pos_integer(), Px::points(), Py::points()) ->
          result() | undefined.

divide_(Size, Px, _Py) when Size < 15 ->
    % For lists smaller than 15 points, brute force is faster.
    brute_force_(Px);
divide_(Size, Px, Py) ->
    LeftSize = Size div 2,
    RightSize = Size - LeftSize,
    {Lx, Rx} = lists:split(LeftSize, Px),
    [{Xref, _} |_] = Rx,
    {Ly, Ry} = split_(Xref, Py),
    LeftResult = divide_(LeftSize, Lx, Ly),
    RightResult = divide_(RightSize, Rx, Ry),
    {MinDist, ClosestPair} = select_(LeftResult, RightResult),
    Xmin = Xref - MinDist,
    Xmax = Xref + MinDist,
    Box = [{X, Y} || {X, Y} <- Py, X > Xmin, X < Xmax],
    conquer_(Box, {MinDist, ClosestPair}).

%% Then conquer the border between divisions
%% The specified points should be sorted on Y and boxed on X.

-spec conquer_(As::points(), Result::result()) -> result().

conquer_([], Result) -> Result;
conquer_([A |As], Result) ->
    % It is proven that only the next 6 points need to be checked
    conquer_(As, A, As, 6, Result).


-spec conquer_(As::points(), A::point(), Bs::points(),
               Depth::pos_integer(), Result::result()) -> result().

conquer_(As, _A, [], _Depth, Result) -> conquer_(As, Result);
conquer_(As, _A, _Bs, 0, Result) -> conquer_(As, Result);
conquer_(As, A, [B |Bs], Depth, Result) ->
    NewResult = select_(Result, {distance_(A, B), {A, B}}),
    conquer_(As, A, Bs, Depth - 1, NewResult).


%% --------------------------------------------------------------------
%% Splits a list in two in function of of given X of reference.
%% The original order is respected and the reference is NOT
%% included in the left side list but on the right side one.

-spec split_(Xref::integer(), Ps::points()) ->
          {points(), points()}.

split_(Xref, Ps) -> split_(Xref, Ps, [], []).


-spec split_(Xref::integer(), Ps::points(), Pl::points(), Pr::points()) ->
          {points(), points()}.

split_(_Xref, [], Pl, Pr) ->
    {lists:reverse(Pl), lists:reverse(Pr)};
split_(Xref, [{X, _Y} = P |Ps], Pl, Pr) when X < Xref ->
    split_(Xref, Ps, [P |Pl], Pr);
split_(Xref, [P |Ps], Pl, Pr) ->
    split_(Xref, Ps, Pl, [P |Pr]).


%% --------------------------------------------------------------------
%% Selects from two results the one with smallest distance.

-spec select_(R1::result(), R2::result() | undefined) -> result().

select_(R1, undefined) -> R1;
select_({D1, _P1} = R1, {D2, _P2}) when D1 < D2 -> R1; 
select_(_R1, R2) -> R2.


%% --------------------------------------------------------------------
%% Calculates the distance between two points.

-spec distance_(A::point(), B::point()) -> float().

distance_({Xa, Ya}, {Xb, Yb}) ->
    math:sqrt(math:pow((Xa - Xb), 2) + math:pow((Ya - Yb), 2)).
    

%% --------------------------------------------------------------------
%% Generates a random set of 2D points with coordinates in
%% range [-65000..65000].

-spec random_data_(Size::pos_integer(), Acc::points()) -> points().

random_data_(0, Acc) -> Acc;
random_data_(Size, Acc) ->
    Point = {random_value_(), random_value_()},
    random_data_(Size - 1, [Point |Acc]).


-spec random_value_() -> integer().

random_value_() ->
    random:uniform(65000*2+1)-65000-1.


%% --------------------------------------------------------------------
%% Benchmarks find_closest/1 with lists of random points of specified
%% size, and print the result to the console.

-spec benchmark_(Fun::function(), Size::pos_integer(), Iter::pos_integer(),
                 Length::pos_or_zero(), Samples::list(pos_or_zero())) -> ok.

benchmark_(_Fun, _Size, Length, Length, Samples) ->
    Min = lists:min(Samples),
    Max = lists:max(Samples),
    Med = lists:nth(round((Length / 2)), lists:sort(Samples)),
    Avg = round(lists:foldl(fun(X, Sum) -> X + Sum end, 0, Samples) / Length),
    Sep = string:copies("*", 21),
    io:format("~s~n" 
              "Min:     ~9b �s~n"
              "Max:     ~9b �s~n"
              "Median:  ~9b �s~n"
              "Average: ~9b �s~n"
              "~s~n", [Sep, Min, Max, Med, Avg, Sep]);
benchmark_(Fun, Size, Iter, Count, Samples) ->
    io:format("Round ~3b...", [Count + 1]),
    Data = random_data(Size),
    {Time, {_Dist, {P1, P2}}} = timer:tc(Fun, [Data]),
    io:format(" ~w ~w~n", [P1, P2]),
    benchmark_(Fun, Size, Iter, Count + 1, [Time |Samples]).


%% --------------------------------------------------------------------
%% Tests
%% --------------------------------------------------------------------

-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").

find_closest_test() ->
    ?assertMatch({_, {{0,0}, {5, 2}}},
                 find_closest([{0,0}, {1,20}, {5, 2}])),
    ?assertMatch({_, {{1,5}, {4, 3}}},
                 find_closest([{-10,10}, {1,5}, {4, 3}])), 
    ?assertMatch({_, {{15, 5}, {17, 7}}},
                 find_closest([{2, 7}, {4, 13}, {5, 7}, {10, 5}, {13, 9},
                               {15, 5}, {17, 7}, {19, 10}, {22, 7}, {25, 10},
                               {29, 14}, {30, 2}])),
     ok.

random_comparison_test() -> test_random_comparisons_(2, 150, 2).

test_random_comparisons_(_, 14, _) -> ok;
test_random_comparisons_(Rep, Size, 0) ->
    test_random_comparisons_(Rep, Size - 1, Rep);
test_random_comparisons_(Rep, Size, Iter) ->
    Data = random_data(Size),
    {_, P1} = brute_force_(Data),
    {_, P2} = find_closest(Data),
    ?assertMatch(true, compare_pairs_(P1, P2)),
    test_random_comparisons_(Rep, Size, Iter - 1).
    
compare_pairs_({A1, A2}, {B1, B2})
  when A1 =:= B1, A2 =:= B2; A1 =:= B2, A2 =:= B1 -> true;
compare_pairs_(_P1, _P2) -> false.
    

-endif.