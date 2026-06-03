import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./en.json";
import zh from "./zh.json";
const saved = (typeof localStorage !== "undefined" && localStorage.getItem("aiws.lang")) || "en";
i18n.use(initReactI18next).init({
  resources: { en: { translation: en }, zh: { translation: zh } },
  lng: saved, fallbackLng: "en",
  // Catalog strings use single-brace placeholders ({v}, {t}, {n}); i18next
  // defaults to double-brace ({{v}}). Match the single-brace convention so
  // interpolation actually fires instead of rendering the literal "{v}".
  interpolation: { escapeValue: false, prefix: "{", suffix: "}" },
});
export default i18n;
