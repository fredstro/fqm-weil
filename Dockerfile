ARG REMOTE_SRC=1
ARG GIT_BRANCH=develop

FROM sagemath/sagemath:latest as base
RUN sudo apt-get update
RUN sudo apt-get -y install git

FROM base as use-git-1
RUN git clone https://github.com/fredstro/fqm-weil.git
WORKDIR "fqm-weil"
RUN git config pull.rebase false && git checkout $GIT_BRANCH

FROM base as use-git-0
ARG GIT_BRANCH=''
COPY --chown=sage . fqm-weil
WORKDIR "fqm-weil"

FROM use-git-${REMOTE_SRC} AS final
RUN sudo apt-get -y install make
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["run"]