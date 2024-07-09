-- +goose Up
-- +goose StatementBegin
CREATE INDEX ON news USING hnsw (embedding vector_cosine_ops);
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin

-- +goose StatementEnd
