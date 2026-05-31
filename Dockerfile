FROM ruby:2.7
ENV RUBYOPT="-W0"
RUN gem install bundler -v 2.4.22
WORKDIR /srv/jekyll
COPY Gemfile .
RUN bundle install
EXPOSE 4000
CMD ["sh", "-c", "bundle install && bundle exec jekyll serve --port 4000 --host 0.0.0.0"]
STOPSIGNAL 2