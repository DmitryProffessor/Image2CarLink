FROM nginx:1.27.2-alpine AS assets
COPY web/index.html /usr/share/nginx/html/index.html

FROM nginxinc/nginx-unprivileged:1.27.2-alpine
COPY --from=assets /usr/share/nginx/html/index.html /usr/share/nginx/html/index.html
EXPOSE 8080
