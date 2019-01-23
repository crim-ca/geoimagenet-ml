#!/usr/bin/env groovy

pipeline {
    agent any

    environment {
        TAG_NAME = sh(returnStdout: true, script: '[[ -z $(git tag -l --points-at HEAD) ]] && printf latest || printf $(git tag -l --points-at HEAD)')
        LOCAL_IMAGE_NAME = "geoimagenet_ml:$TAG_NAME"
        LATEST_IMAGE_NAME = "docker-registry.crim.ca/geoimagenet/ml:latest"
        TAGGED_IMAGE_NAME = "docker-registry.crim.ca/geoimagenet/ml:$TAG_NAME"
    }

    options {
        buildDiscarder (logRotator(numToKeepStr:'10'))
    }

    stages {

        stage('Build') {
            steps {
                sh 'env | sort'
                git branch: 'master',
                     credentialsId: 'f6c3d8c2-ac53-45bd-971e-1a3a02da3b19',
                     changelog: false,
                     poll: false,
                     url: 'https://www.crim.ca/stash/scm/VISI/thelper.git'
                sh 'DOCKER_REPO=$LOCAL_IMAGE_NAME make docker-build -f geoimagenet_ml/Makefile'
            }
        }

        stage('Test') {
            steps {
                script {
                    docker.image('mongo:3.4.0').withRun('-e "ALLOW_IP_RANGE=0.0.0.0/0" -e "IP_LIST=*"') { c ->
                        sh """
                        docker run --rm --link ${c.id}:mongodb -e MONGODB_HOST=mongodb $LOCAL_IMAGE_NAME /bin/sh -c \" \
                        pip install -r requirements-dev.txt && \
                        pytest -v\"
                        """
                    }
                }
            }
        }

        stage('Deploy') {
            when {
                environment name: 'GIT_LOCAL_BRANCH', value: 'release'
            }
            steps {
                sh 'docker tag $LOCAL_IMAGE_NAME $TAGGED_IMAGE_NAME'
                sh 'docker push $TAGGED_IMAGE_NAME'
                sh 'docker tag $LOCAL_IMAGE_NAME $LATEST_IMAGE_NAME'
                sh 'docker push $LATEST_IMAGE_NAME'
                sh 'ssh ubuntu@geoimagenetdev.crim.ca "cd ~/compose && ./geoimagenet-compose.sh down && ./geoimagenet-compose.sh pull && ./geoimagenet-compose.sh up -d"'
                slackSend channel: '#geoimagenet', color: 'good', message: "*GeoImageNet ML*:\nPushed docker image: `${env.TAGGED_IMAGE_NAME}`\nDeployed to: https://geoimagenetdev.crim.ca/ml"
            }
        }
    }
    post {
       success {
           slackSend channel: '#geoimagenet', color: 'good', message: "*GeoImageNet ML*: Build #${env.BUILD_NUMBER} *successful* on git branch `${env.GIT_LOCAL_BRANCH}` :tada: (<${env.BUILD_URL}|View>)"
       }
       failure {
           slackSend channel: '#geoimagenet', color: 'danger', message: "*GeoImageNet ML*: Build #${env.BUILD_NUMBER} *failed* on git branch `${env.GIT_LOCAL_BRANCH}` :sweat_smile: (<${env.BUILD_URL}|View>)"
       }
    }
}
