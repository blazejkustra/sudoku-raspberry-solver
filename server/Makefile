.DEFAULT_GOAL := run

run:
				@if [ ! -f ./.env ]; then cp .env.example .env; fi;
				@python3 src/manage.py run

test:
				@python3 src/manage.py test
