-- +goose Up
-- +goose StatementBegin
CREATE EXTENSION vector;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP EXTENSION vector;
-- +goose StatementEnd
