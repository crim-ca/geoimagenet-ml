<!DOCTYPE html>
<head>
    <meta charset="UTF-8">
    <title>${api_title}</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.17.5/swagger-ui.css">
    <script src="https://unpkg.com/swagger-ui-dist@3.17.5/swagger-ui-standalone-preset.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@3.17.5/swagger-ui-bundle.js"></script>
    <script>
        addEventListener('DOMContentLoaded', function() {
            window.ui = SwaggerUIBundle({
                url: "${api_swagger_json_path}",
                urls: [{ url: "${api_swagger_json_path}", name: "latest" }],
                dom_id: '#swagger-ui',
                deepLinking: true,
                docExpansion: 'none',
                validatorUrl: false,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                tagsSorter: 'alpha',
                apisSorter : "alpha",
                operationsSorter: "alpha",
            });
        });
    </script>
</head>
<body>
<div id="swagger-ui"></div>
</body>
</html>
