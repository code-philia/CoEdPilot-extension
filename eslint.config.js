export default [
  {
    files: ["**/*.js", "**/*.ts"],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: "module",
    },
    rules: {
      "semi": ["error", "always"],
      "quotes": ["error", "double"],
      // 添加更多规则
    },
  },
];