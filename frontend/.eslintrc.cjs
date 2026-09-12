module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    "eslint:recommended",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended",
  ],
  parserOptions: { ecmaVersion: "latest", sourceType: "module" },
  settings: { react: { version: "detect" } },
  rules: {
    "react/react-in-jsx-scope": "off",   // not needed with Vite
    "react/prop-types": "off",           // project uses runtime checks, not PropTypes
  },
};
