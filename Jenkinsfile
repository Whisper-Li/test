Jenkinsfile (Declarative Pipeline)
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                echo 'yes'
            }
        }
    }
    post {
        always {
            junit 'build/reports/1/1.xml'
        }
    }
}
