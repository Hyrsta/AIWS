import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./en.json";
import zh from "./zh.json";
const saved = (typeof localStorage !== "undefined" && localStorage.getItem("aiws.lang")) || "en";
i18n.use(initReactI18next).init({
  resources: { en: { translation: en }, zh: { translation: zh } },
  lng: saved, fallbackLng: "en", interpolation: { escapeValue: false },
});
export default i18n;
