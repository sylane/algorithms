REBAR=`which rebar || ./rebar`
RELOADER=`if erl -noshell -eval 'R = case code:load_file(reloader) of {module, _} -> init:stop(0); _ -> init:stop(1) end.'; then echo "-s reloader"; fi`

all: deps compile

deps:
	@$(REBAR) get-deps

compile:
	@$(REBAR) compile

test:
	@$(REBAR) skip_deps=true eunit

clean:
	@$(REBAR) clean

shell: deps compile
	@erl -pz ebin $(RELOADER)
